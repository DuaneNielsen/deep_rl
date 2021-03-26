import buffer as bf
import wandb
import checkpoint
import driver
from statistics import mean, stdev
from pathlib import Path
from torchvision.io import write_video
import torch
import numpy as np
import time


def _render_env(env, render, delay):
    if render:
        env.render()
        time.sleep(delay)


class Evaluator:
    """
    Evaluates policies and saves them if they improve

    Args:
        make_env: make_env() -> env a function that returns the environment
        seed: random seed
        plot: plot the episode returns if True
    """
    def __init__(self, env):
        self.best_mean_return = -999999999
        env, buffer = bf.wrap(env)
        self.env = env
        self.buffer = buffer
        self.vid_buffer = []

    def write_mp4(self, file):
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.vid_buffer))
        write_video(str(file), stream, 24.0)
        del self.vid_buffer
        self.vid_buffer = []

    def clear_vid_buffers(self):
        del self.vid_buffer
        self.vid_buffer = []

    def episode(self, env, policy, render=False, delay=0.01, capture=False):
        """
        Runs one episode using the provided policy on the environment

        Args:
            env: gym environment to generate an episode for
            policy: policy(state) -> action takes state as input, must output an action runnable on the environment
            render: if True will call environments render function
            delay: rendering delay
            kwargs: kwargs will be passed to policy
        """

        returns = 0
        length = 0

        with torch.no_grad():
            state, reward, done, info = env.reset(), 0.0, False, {}
            action = policy(state)
            _render_env(env, render, delay)

            if capture:
                frame = self.env.render(mode='rgb_array')
                self.vid_buffer.append(frame)

            while not done:
                state, reward, done, info = env.step(action)
                returns += reward
                length += 1
                action = policy(state)
                _render_env(env, render, delay)
                if capture:
                    frame = self.env.render(mode='rgb_array')
                    self.vid_buffer.append(frame)
                if done:
                    wandb.log({'test_reward': returns, 'test_len': length})

    def sample_policy_returns(self, policy, samples, render=False, capture=False):
        """
        samples n trajectories from environment using policy

        Args:
            policy: policy(state) -> action, the policy to evaluate
            samples: number of episodes to sample
            render: render while running
            capture: capture rgb render to buffer
        Returns:
             mean_return
             stdev_return
        """
        start = len(self.buffer.trajectories)
        # run test trajectories and compute the returns
        for _ in range(samples):
            self.episode(self.env, policy, render=render, capture=capture)
        returns = [traj_info['return'] for traj_info in self.env.trajectory_info[start:]]
        mean_return = mean(returns)
        stdev_return = stdev(returns)
        self.buffer.clear()
        return mean_return, stdev_return

    def evaluate(self, policy, run_dir, params, sample_n=10, render=False, capture=False):
        """
        Evaluate the policy and save if improved

        Args:
            policy: policy(state) -> action to evaluate
            run_dir: directory to save parameters to
            params: dict of parameters to write..

            .. code-block:: python
              policy_net = nn.Linear(4, 1)
              optim = nn.optim.SGD(policy_net.parameters(), lr=1e-4)
              params={'policy_net': policy_net, 'optim': optim}

            sample_n: number of episodes to sample
            render: render the environment while running if True

        Returns:
            mean_return
            stdev_return

        """
        mean_return, stdev_return = self.sample_policy_returns(policy, sample_n, render, capture=capture)

        wandb.run.summary["last_mean_return"] = mean_return
        wandb.run.summary["last_stdev_return"] = stdev_return

        # checkpoint policy if mean return is better
        if mean_return > self.best_mean_return:
            best_mean_return = mean_return
            wandb.run.summary["best_mean_return"] = best_mean_return
            wandb.run.summary["best_stdev_return"] = stdev_return
            checkpoint.save(run_dir, 'best', **params)
            if capture:
                self.write_mp4(f'{run_dir}/best_eps.mp4')
                wandb.log({"video": wandb.Video(f'{run_dir}/best_eps.mp4', fps=4, format="gif")})

        self.clear_vid_buffers()

        return mean_return, stdev_return

    def demo(self, demo, policy):
        """
        Infinite loop that runs policy on environment, and renders
        Args:
            demo: if true enters infinite loop
            policy: policy(state) -> action to run

        """
        if demo:
            while True:
                driver.episode(self.env, policy, render=True)
                self.buffer.clear()