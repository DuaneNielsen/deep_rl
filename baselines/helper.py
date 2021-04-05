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


def write_mp4(file, vid_buffer):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    stream = torch.from_numpy(np.stack(vid_buffer))
    write_video(str(file), stream, 24.0)


def demo(demo, env, policy):
    """
    Infinite loop that runs policy on environment, and renders
    Args:
        demo: if true enters infinite loop
        policy: policy(state) -> action to run

    """
    if demo:
        while True:
            driver.episode(env, policy, render=True)


def episode(env, policy, render=False, delay=0.01, capture=False):
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
    vid_buffer = []

    with torch.no_grad():
        state, reward, done, info = env.reset(), 0.0, False, {}
        action = policy(state)
        _render_env(env, render, delay)

        if capture:
            frame = env.render(mode='rgb_array')
            vid_buffer.append(frame)

        while not done:
            state, reward, done, info = env.step(action)
            returns += reward
            length += 1
            action = policy(state)
            _render_env(env, render, delay)
            if capture:
                frame = env.render(mode='rgb_array')
                vid_buffer.append(frame)

    return returns, length, vid_buffer


class Evaluator:
    """
    Evaluates policies and saves them if they improve

    Args:
        make_env: make_env() -> env a function that returns the environment
        seed: random seed
        plot: plot the episode returns if True
    """
    def __init__(self):
        self.best_mean_return = -999999999.0
        self.test_number = 0

    def evaluate(self, env, policy, run_dir, params, sample_n=10, render=False, capture=False, global_step=None):
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
        returns = []
        vidstream = []

        for _ in range(sample_n):
            retn, length, video = episode(env, policy, render=render, capture=capture)
            returns.append(retn)
            if capture:
                vidstream.extend(video)

        mean_return = mean(returns)
        stdev_return = stdev(returns)
        wandb.run.summary["last_mean_return"] = mean_return
        wandb.run.summary["last_stdev_return"] = stdev_return

        wandb.log({"test_returns": wandb.Histogram(returns),
                   "test_mean_return": mean_return,
                   "global_step": global_step,
                   "test_number": self.test_number})
        self.test_number += 1

        # checkpoint policy if mean return is better
        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            wandb.run.summary["best_mean_return"] = self.best_mean_return
            wandb.run.summary["best_stdev_return"] = stdev_return
            checkpoint.save(run_dir, 'best', **params)

        if capture:
            write_mp4(f'{run_dir}/test_run_{self.test_number}.mp4', vidstream)
            wandb.log({"video": wandb.Video(f'{run_dir}/test_run_{self.test_number}.mp4', fps=4, format="gif")})

        return mean_return, stdev_return

