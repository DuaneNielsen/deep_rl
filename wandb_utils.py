import gym
import wandb
import torch


class LogRewards(gym.Wrapper):
    """
    Wrapper to log episode rewards and lengths for wandb

    Args:
        prefix: prefix to prepend to the metric names
    """
    def __init__(self, env, prefix=None):
        super().__init__(env)
        self.reward = 0
        self.len = 0
        self.run_steps = 0
        self.prefix = prefix + '_' if prefix is not None else ''

    def reset(self):
        """ wraps the env reset method """
        self.reward = 0
        self.len = 0
        return self.env.reset()

    def step(self, action):
        """ wraps the env step method """
        state, reward, done, info = self.env.step(action)
        self.reward += reward
        self.len += 1
        self.run_steps += 1
        if done:
            wandb.log({
                f'{self.prefix}epi_reward': self.reward,
                f'{self.prefix}epi_len': self.len}, step=self.run_steps)
        return state, reward, done, info


def nancheck(tensor, error):
    """ checks the tensor for nan and reports error to wandb if detected. then throws assertion """
    if torch.isnan(tensor).any():
        wandb.summary['FAIL'] = error
        assert False, error

