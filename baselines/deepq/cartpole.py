import torch
from torch import nn
import buffer as bf
import gym
import random
import algos.dqn as dqn
import driver
from gymviz import Plot

if __name__ == '__main__':

    """ configuration """
    batch_size = 8
    epsilon = 0.1
    discount = 0.9
    hidden_size = 12

    """ Environment """
    env = gym.make('CartPole-v1')

    """ replay buffer """
    env, replay_buffer = bf.wrap(env)
    env = Plot(env, blocksize=1)

    """ check environment has continuous input, discrete output"""
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert len(env.observation_space.shape) == 1

    class QNet(nn.Module):
        """
        Simple MLP, takes in state and outputs a value for each action
        """
        def __init__(self, state_size, hidden_size, action_size):
            super().__init__()
            self.q_net = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, action_size))

        def forward(self, state):
            action = self.q_net(state)
            return action

    q_net = QNet(state_size=env.observation_space.shape[0], hidden_size=12, action_size=env.action_space.n)
    optim = torch.optim.SGD(q_net.parameters(), lr=1e-3)

    """ epsilon greedy policy to run on environment """
    def policy(state):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(q_net(state), dim=1)
            return action.item()

    """ execute 1 transition on environment """
    for step_n, _ in enumerate(driver.step_environment(env, policy, render=True)):
        if step_n < batch_size:
            continue
        if step_n > 30000:
            break
        """ sample a batch and update """
        dqn.train(replay_buffer, q_net, optim, batch_size, discount=discount)
