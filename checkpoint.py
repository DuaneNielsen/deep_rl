import buffer as bf
import driver
from statistics import mean, stdev
from pathlib import Path
import torch


def sample_policy_returns(env, policy, samples, render=False):
    """
    samples n trajectories from environment using policy
    returns the mean_return and stdev_return
    """
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


def save(directory, prefix=None, **kwargs):
    """
    save torch.nn Modules to a directory, uses the state_dict() method

    directory: the directory to save to, will be created if it doesnt exist
    prefix: prefix to apply to the files to be saved
    kwargs: argument name is taken as the save file name,
        argument value is a torch.nn.Module to be saved

        eg: checkpoint.save('runs/run_42', 'best', policy=policy_net, optim=optim)
        will write out files
            runs/run_42/best_policy.sd
            runs/run_42/best_optim.sd

    """
    prefix = prefix + '_' if prefix is not None else ''
    Path(directory).mkdir(parents=True, exist_ok=True)
    for key, net in kwargs.items():
        torch.save(net.state_dict(), directory + '/' + prefix + key + '.sd')


def load(directory, prefix=None, **kwargs):
    """
    loads saved weights into a torch.nn.Module using the state_dict() method
    directory: the directory to load from
    prefix: a prefix associated with the files
    kwargs: argument name is take to be the file to look for,
        argument value is a torch.nn.Module to load

        eg checkpoint.load('runs/run_42', 'best', policy=policy_net, optim=optim)

        will load the state_dicts from disk for policy_net and optim

    """
    sd = {}
    prefix = prefix + '_' if prefix is not None else ''
    for file in Path(directory).glob(f'{prefix}*.sd'):
        key = file.name[len(prefix):-3]
        sd[key] = file

    for key, net in kwargs.items():
        assert key in sd, f"could not find a file for {key}"
        net.load_state_dict(torch.load(sd[key]))


