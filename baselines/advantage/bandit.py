import torch
import torch.nn as nn

from env import debug
import buffer as bf
import random
import algos.advantage as adv
import time

if __name__ == '__main__':

    """ environment
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), S, T(1.0)]
    """
    env = debug.Bandit()
    env, buffer = bf.wrap(env, plot=True, plot_blocksize=16)

    """ configuration """
    epsilon = 0.05  # exploration parameter, prob of taking random action
    batch_size = 8
    discount = 1.0

    """ this Value network """
    class VNet(nn.Module):
        def __init__(self):
            super(VNet, self).__init__()
            self.vtable = nn.Parameter(torch.randn(3, 1))

        def forward(self, state):
            s = torch.argmax(state, dim=1)
            return self.vtable[s]

    q_net = VNet()
    optim = torch.optim.SGD(q_net.parameters(), lr=1e-2)

    """ policy to run on environment """
    def policy(state):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(q_net(state), dim=1)
            return action.item()

    """
    training loop  
    each iteration generates 1 transition on the environment and adds to to replay buffer 
    """
    for step_n, (s, a, s_p, r, d, i) in enumerate(bf.step_environment(env, policy, render=False)):
        if step_n < batch_size:
            continue
        if step_n > 2000:
            break

        """ train the q network using dqn """
        dqn.train(buffer, q_net, optim, batch_size=batch_size, discount=discount)

        """ periodically print the Q table to the console """
        if step_n % 100 == 0:
            print(q_net.qtable.T)
        time.sleep(0.01)




