import torch
import torch.nn as nn

from env import trivial
import buffer as bf
import random
import algos.dqn as dqn
import time

if __name__ == '__main__':
    env = trivial.Bandit()
    env, buffer = bf.wrap(env, plot=True, plot_blocksize=16)

    eps = 0.05


    class QNet(nn.Module):
        def __init__(self):
            super(QNet, self).__init__()
            self.qtable = nn.Parameter(torch.randn(3, 2))

        def forward(self, state):
            s = torch.argmax(state, dim=1)
            return self.qtable[s]

    q_net = QNet()
    optim = torch.optim.SGD(q_net.parameters(), lr=1e-2)


    def policy(state):
        if random.random() < eps:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(q_net(state), dim=1)
            return action.item()

    batch_size = 8

    for step_n, (s, s_i, a, s_p, r, d, i) in enumerate(bf.transitions(env, policy, render=True)):
        if step_n < batch_size:
            continue
        if step_n > 2000:
            break
        dqn.train(buffer, q_net, optim, batch_size, discount=1.0)
        time.sleep(0.01)




