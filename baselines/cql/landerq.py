import torch
from torch import nn
import gym
from argparse import ArgumentParser
import rl
from gym.wrappers.transform_reward import TransformReward
import torch_utils
import wandb_utils
import wandb
from rich.progress import Progress


def rescale_reward(reward):
    return reward * config.env_reward_scale - config.env_reward_bias


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--video_episodes', type=int, default=10)
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--test_episodes', type=int, default=16)
    parser.add_argument('--demo', action='store_true', default=False)
    config = parser.parse_args()

    wandb.init(project=f"cql-LunarLander-v2", config=config, tags=['logging_policy'])

    """ Environment """
    def make_env():
        env = gym.make('LunarLander-v2')
        env = TransformReward(env, rescale_reward)
        """ check environment has continuous input, discrete output"""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 1
        assert isinstance(env.action_space, gym.spaces.Discrete)
        return env

    env = make_env()
    test_env = make_env()
    buffer = rl.ReplayBuffer()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        env.seed(config.seed)
        env.action_space.seed(config.seed)

    """ network """
    class LanderNet(nn.Module):
        """
        Simple MLP, takes in state and outputs a value for each action
        """

        def __init__(self, state_size, hidden_size, action_size):
            super().__init__()
            """ hardcoded the normalization for Lunar Lander """
            self.u_s = torch.tensor([0.0034, 0.9705, 0.0122, -0.6750, -0.0066, -0.0051, 0.0130, 0.0141])
            self.std_s = torch.tensor([0.2931, 0.4605, 0.6129, 0.4766, 0.5026, 0.4014, 0.1135, 0.1181])
            self.u_v = torch.tensor([9.1955e-02, 3.0285e-05, -8.5671e-02, -5.2343e-03])
            self.std_v = torch.tensor([0.0973, 0.2776, 0.3580, 0.2218])

            self.q_net1 = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(inplace=True),
                                        nn.Linear(hidden_size, action_size))

            self.q_net2 = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(inplace=True),
                                        nn.Linear(hidden_size, action_size))

        def forward(self, state):
            s = (state - self.u_s) / self.std_s  # normalize
            v1 = self.q_net1(s)
            v2 = self.q_net2(s)
            v1 = (v1 - self.u_v) / self.std_v  # normalize
            v2 = (v2 - self.u_v) / self.std_v  # normalize
            value, _ = torch.stack([v1, v2], dim=-1).min(-1)
            return value


    q_net = LanderNet(state_size=env.observation_space.shape[0], hidden_size=16, action_size=env.action_space.n)

    """ load weights from file if required"""
    q_net.load_state_dict(torch.load('lander_q_net.sd'))

    def exploit_policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.argmax(q_net(state), dim=1)
        return action.item()

    episodes_captured = 0
    test_number = 1
    vidstream = []

    """ demo """
    with Progress() as progress:
        run = progress.add_task('Generating', total=config.episodes)
        for step, s, a, s_p, r, d, i, m in rl.step(env, exploit_policy, buffer, render=True):
            if episodes_captured < config.episodes:
                buffer.append(s, a, s_p, r, d)
                if episodes_captured < config.video_episodes:
                    vidstream.append(m['frame'])
                else:
                    rl.global_render = False
                episodes_captured += 1 if d else 0
                progress.update(run, advance=1 if d else 0)
            else:
                break
            """ test """
            if step > config.test_steps * test_number:
                stats = rl.evaluate(test_env, exploit_policy, sample_n=config.test_episodes)

                vid_filename = None
                if 'video' in stats:
                    vid_filename = f'{config.run_dir}/test_run_{test_number}.mp4'
                    torch_utils.write_mp4(vid_filename, stats['video'])
                wandb_utils.log_test_stats(stats, vid_filename)

                test_number += 1

    video_filename = f'./lander_{len(buffer)}.mp4'
    torch_utils.write_mp4(video_filename, vidstream)
    wandb.log({'video': wandb.Video(video_filename, fps=4, format="mp4")})

    if not config.demo:
        filename = f'./lander_{len(buffer)}.pkl'
        rl.save(buffer, filename)
        wandb.run.tags = [*wandb.run.tags, filename]
