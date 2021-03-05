import gym
import wandb


class LogRewards(gym.Wrapper):
    """
    Wrapper to log episode rewards for wandb
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward = 0
        self.len = 0

    def reset(self):
        self.reward = 0
        self.len = 0
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.reward += reward
        self.len += 1
        if done:
            wandb.log({'epi_reward': self.reward, 'epi_len': self.len})
        return state, reward, done, info

