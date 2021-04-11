import gym
import wandb
import torch
import time
from statistics import mean, stdev
import checkpoint
from pathlib import Path
from torchvision.io import write_video
import numpy as np


class LogRewards(gym.Wrapper):
    """
    Wrapper to log episode rewards and lengths for wandb

    Args:
        prefix: prefix to prepend to the metric names
    """
    def __init__(self, env, prefix=None):
        super().__init__(env)
        self.prev_reward = 0
        self.prev_len = 0
        self.reward = 0
        self.len = 0
        self.prefix = prefix + '_' if prefix is not None else ''
        self.global_step = None

    def reset(self):
        """ wraps the env reset method """
        self.prev_reward = self.reward
        self.prev_len= self.len
        self.reward = 0
        self.len = 0
        return self.env.reset()

    def step(self, action):
        """ wraps the env step method """
        state, reward, done, info = self.env.step(action)
        self.reward += reward
        self.len += 1
        wandb.log({
            f'{self.prefix}epi_reward': self.prev_reward,
            f'{self.prefix}epi_len': self.prev_len,
            'global_step': self.global_step})
        return state, reward, done, info


def nancheck(tensor, error):
    """ checks the tensor for nan and reports error to wandb if detected. then throws assertion """
    if torch.isnan(tensor).any():
        wandb.summary['FAIL'] = error
        assert False, error


global_step = 0


def step_environment(env, policy, render=False, timing=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration

    Args:
        env: gym environment to step
        policy: policy to use, policy(state) -> action
        render: calls env render function if True
        timing: prints timing info to stdout if True
        kwargs: will be passed to the policy
    """
    global global_step
    step_time, render_time, policy_time = [], [], []
    step_t, start_t, policy_t, render_t = 0, 0, 0, 0
    done = True
    state = None
    epi_returns = 0
    epi_len = 0
    global_step = 0

    while True:

        if timing:
            start_t = time.time()

        if done:
            state, state_info = env.reset(), {}
            wandb.log({'epi_returns': epi_returns, 'epi_len': epi_len}, step=global_step)
            epi_returns = 0
            epi_len = 0
            if render:
                env.render()

        action = policy(state, **kwargs)

        if timing:
            policy_t = time.time()

        state_p, reward, done, info = env.step(action)

        if timing:
            step_t = time.time()

        if render:
            env.render()

        if timing:
            render_t = time.time()

        yield state, action, state_p, reward, done, info

        state = state_p
        done = done
        global_step += 1
        epi_len += 1
        epi_returns += reward

        policy_time += [policy_t - start_t]
        step_time += [step_t - policy_t]
        render_time += [render_t - step_t]

        if global_step % 1000 == 0:
            perf_log = {}
            mean_policy_time = mean(policy_time) * 1000
            mean_step_time = mean(step_time) * 1000
            mean_render_time = mean(render_time) * 1000

            if mean_render_time > 1.0:
                perf_log['render_time (ms)'] = mean_render_time
            if mean_step_time > 1.0:
                perf_log['step_time (ms)'] = mean_step_time
            if mean_policy_time > 1.0:
                perf_log['policy_time (ms)'] = mean_policy_time

            if len(perf_log) > 0:
                wandb.log(perf_log, step=global_step)

            step_time, render_time, policy_time = [], [], []


def _render_env(env, render, delay):
    if render:
        env.render()
        time.sleep(delay)


def write_mp4(file, vid_buffer):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    stream = torch.from_numpy(np.stack(vid_buffer))
    write_video(str(file), stream, 24.0)


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

    if render:
        env.render()

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


def demo(demo, env, policy):
    """
    Infinite loop that runs policy on environment, and renders
    Args:
        demo: if true enters infinite loop
        policy: policy(state) -> action to run

    """
    if demo:
        while True:
            episode(env, policy, render=True)


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

    def evaluate_now(self, test_steps):
        """
        Convenience method, returns true when time to run a test
        Args:
            global_step: the amount of steps run
            test_steps: the amount of steps to run before testing

        Returns: true when it's time to run a test

        """
        global global_step
        return global_step > test_steps * (self.test_number + 1)

    def evaluate(self, env, policy, run_dir, params, sample_n=10, render=False, capture=False):
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
        global global_step
        start_t = time.time()

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
        global_step += 2

        # checkpoint policy if mean return is better
        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            wandb.run.summary["best_mean_return"] = self.best_mean_return
            wandb.run.summary["best_stdev_return"] = stdev_return
            checkpoint.save(run_dir, 'best', **params)

        if capture:
            write_mp4(f'{run_dir}/test_run_{self.test_number}.mp4', vidstream)
            wandb.log({"video": wandb.Video(f'{run_dir}/test_run_{self.test_number}.mp4', fps=4, format="gif")})

        end_t = time.time()
        total_t = end_t - start_t

        wandb.log({"test_returns": wandb.Histogram(returns),
                   "test_mean_return": mean_return,
                   "test_wall_time": total_t}, step=global_step)

        self.test_number += 1

        return mean_return, stdev_return

