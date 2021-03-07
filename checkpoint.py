import buffer as bf
import driver
from statistics import mean, stdev
from pathlib import Path
import torch


def sample_policy_returns(env, policy, samples, render=False):
    env, buffer = bf.wrap(env)
    start = len(buffer.trajectories)
    # run test trajectories and compute the returns
    for _ in range(samples):
        driver.episode(env, policy, render=render)
    returns = [traj_info['return'] for traj_info in env.trajectory_info[start:]]
    mean_return = mean(returns)
    stdev_return = stdev(returns)
    buffer.clear()
    return mean_return, stdev_return


def save(dir, prefix=None, **kwargs):
    prefix = prefix + '_' if prefix is not None else ''
    Path(dir).mkdir(parents=True, exist_ok=True)
    for key, net in kwargs.items():
        torch.save(net.state_dict(), dir + '/' + prefix + key + '.sd')


def load(dir, prefix=None, **kwargs):
    sd = {}
    prefix = prefix + '_' if prefix is not None else ''
    for file in Path(dir).glob(f'{prefix}*.sd'):
        key = file.name[len(prefix):-3]
        sd[key] = file

    for key, net in kwargs.items():
        assert key in sd, f"could not find a file for {key}"
        net.load_state_dict(torch.load(sd[key]))


