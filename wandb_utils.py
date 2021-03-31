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


