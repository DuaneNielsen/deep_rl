import torch
import torch.nn as nn

from env import debug
import buffer as bf
import random
import numpy as np
import algos.advantage as adv
import time
from gym import wrappers


if __name__ == '__main__':

    """ environment 
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), E, E, S, E, E, T(1.0)]
    """
    unwrapped = debug.DelayedBandit()
    env = wrappers.TimeLimit(unwrapped, max_episode_steps=100)
    env, buffer = bf.wrap(env, plot=True, plot_blocksize=16)

    """ configuration """
    epsilon = 0.05  # exploration parameter, prob of taking random action
    batch_size = 8
    discount = 1.0

    """ Value network """
    class VNet(nn.Module):
        def __init__(self):
            super(VNet, self).__init__()
            self.vtable = nn.Parameter(torch.randn(7, 1))

        def forward(self, state):
            s = torch.argmax(state, dim=1)
            return self.vtable[s]

    v_net = VNet()
    optim = torch.optim.SGD(v_net.parameters(), lr=1e-2)

    """ policy to run on environment """
    def policy(state):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            next_state = []
            reward = []
            for action_candidate in range(env.action_space.n):
                s_p, r, d, i = unwrapped.lookahead(state, action_candidate)
                next_state += [s_p]
                reward += [r]
            next_state = np.stack(next_state)
            reward = np.stack(reward)
            next_state = torch.from_numpy(next_state).float()
            reward = torch.from_numpy(reward).float().unsqueeze(1)
            values = v_net(next_state) + reward
            return torch.argmax(values, dim=0).item()


    """
    training loop  
    each iteration generates 1 transition on the environment and adds to to replay buffer 
    """
    for step_n, (s, a, s_p, r, d, i) in enumerate(bf.step_environment(env, policy, render=False)):
        if step_n < batch_size:
            continue
        if step_n > 2000000:
            break

        """ train the q network using dqn """
        adv.train(buffer, v_net, optim, batch_size=batch_size, discount=discount)

        """ periodically print the Q table to the console """
        if step_n % 100 == 0:
            print(v_net.vtable.T)
        time.sleep(0.01)




