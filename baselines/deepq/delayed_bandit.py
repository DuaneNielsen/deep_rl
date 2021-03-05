import torch
import torch.nn as nn

import driver
import gym
from env import wrappers
from gymviz import Plot
import buffer as bf
import random
import algos.dqn as dqn
import time

if __name__ == '__main__':

    """ environment 
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), E, E, S, E, E, T(1.0)]
    """
    env = gym.make('DelayedBandit-v1')
    env = wrappers.TimeLimit(env, max_episode_steps=10)
    env, buffer = bf.wrap(env)
    env = Plot(env, episodes_per_point=16)

    """ configuration """
    epsilon = 0.05
    batch_size = 8

    class QNet(nn.Module):
        """
        simple Q lookup table: S x A
        """
        def __init__(self):
            super(QNet, self).__init__()
            self.qtable = nn.Parameter(torch.randn(7, 2))

        def forward(self, state):
            """
            param: state N, S
            returns: values for each action N, A
            """
            s = torch.argmax(state, dim=1)
            return self.qtable[s]

    q_net = QNet()
    optim = torch.optim.SGD(q_net.parameters(), lr=1e-1)

    """ eplison greedy policy to run on environment """
    def policy(state):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(q_net(state), dim=1)
            return action.item()


    """ 
    steps the environment 1 step using policy
    samples batch and applies 1 dqn update   
    """
    for step_n, _ in enumerate(driver.step_environment(env, policy, render=False)):
        if step_n < batch_size:
            continue
        if step_n > 20000:
            break
        dqn.train(buffer, q_net, optim, batch_size=batch_size, discount=1.0)
        if step_n % 100 == 0:
            print(q_net.qtable.T)
        time.sleep(0.01)




