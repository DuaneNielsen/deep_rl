import torch
import torch.nn as nn

from env import trivial
import buffer as bf
import random
import algos.deepq as dqn
import time

if __name__ == '__main__':
    env = trivial.Bandit()
    env, buffer = bf.wrap(env, plot=True, plot_blocksize=16)

    eps = 0.05

    class QNet(nn.Module):
        def __init__(self):
            super(QNet, self).__init__()
            """
            init the left and right values to the wrong answers
            """
            self.left = nn.Parameter(torch.tensor([1.0]))
            self.right = nn.Parameter(torch.tensor([-1.0]))

        def forward(self, state, action):
            N = action.shape[0]
            values = torch.cat([self.left, self.right]).reshape(1, -1).repeat_interleave(N, dim=0)
            return torch.sum(values * action, dim=1)

    q_net = QNet()
    optim = torch.optim.SGD(q_net.parameters(), lr=1e-2)


    def policy(state):
        if random.random() < eps:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            value, action = dqn.max_q(q_net, state, 2)
            return action.item()

    batch_size = 8

    for step_n, (s, s_i, a, s_p, r, d, i) in enumerate(bf.transitions(env, policy, render=True)):
        if step_n < batch_size:
            continue
        if step_n > 2000:
            break
        dqn.train(buffer, q_net, optim, batch_size, 2, 1.0)
        time.sleep(0.01)




