import torch
import torch.nn as nn

import gym
import env

from gymviz import Plot

from algos import ppo
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser
import rl
import checkpoint
from rich.progress import Progress
import torch_utils
import wandb_utils
import wandb


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--log_episodes', type=int, default=0)
    parser.add_argument('--video_episodes', type=int, default=3)
    parser.add_argument('--test_episodes', type=int, default=8)
    parser.add_argument('--test_steps', type=int, default=1000)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    """ environment """
    def make_env():
        env = gym.make('Pendulum-v0')
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ test env """
    env = make_env()
    if config.plot:
        env = Plot(env, episodes_per_point=1, title=f'ppo-a2c-Pendulum-v0')

    """ network """
    class A2CNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self, input_dims, hidden_dims, min, max):
            super().__init__()
            self.min = min
            self.max = max
            self.mu = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, 2))
            self.scale = nn.Linear(input_dims, 1, bias=False)

        def forward(self, state):
            output = self.mu(state)
            value = output[..., 0:1]
            mu = output[..., 1:2]
            scale = torch.sigmoid(self.scale(state)) + 0.01
            a_dist = ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)
            return value, a_dist

    a2c_net = A2CNet(
        input_dims=env.observation_space.shape[0],
        hidden_dims=16,
        min=env.action_space.low[0],
        max=env.action_space.high[0])

    a2c_net = ppo.PPOWrapModel(a2c_net)

    checkpoint.load('.', prefix='pendulum', a2c_net=a2c_net)

    """ policy to run on environment """
    def policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            value, action = a2c_net(state)
            a = action.rsample()
            assert torch.isnan(a) == False
            return a.numpy()

    if config.log_episodes == 0:
        """ demo  (infinite loop) """
        rl.demo(True, env, policy)
    else:
        """ log mode """
        wandb.init(project=f"cql-Pendulum-v0", config=config)

        buffer = rl.ReplayBuffer()
        episodes_captured = 0
        test_number = 0
        vidstream = []

        with Progress() as progress:
            run = progress.add_task('Generating', total=config.log_episodes)
            for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer, render=True):

                if episodes_captured < config.log_episodes:
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
                    stats = rl.evaluate(env, policy, sample_n=config.test_episodes)

                    vid_filename = None
                    if 'video' in stats:
                        vid_filename = f'{config.run_dir}/test_run_{test_number}.mp4'
                        torch_utils.write_mp4(vid_filename, stats['video'])
                    wandb_utils.log_test_stats(stats, vid_filename)

                    test_number += 1

        """ log transitions """
        filename = f'./Pendulum-v0_{len(buffer)}.pkl'
        rl.save(buffer, filename)
        wandb.run.tags = [*wandb.run.tags, filename]

        """ log video """
        video_filename = f'./Pendulum-v0_{len(buffer)}.mp4'
        torch_utils.write_mp4(video_filename, vidstream)
        wandb.log({'video': wandb.Video(video_filename, fps=4, format="mp4")})
