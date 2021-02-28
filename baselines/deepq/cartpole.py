import torch
from torch import nn
import buffer as bf
import gym
import random
from algos import deepq
from algos.deepq import max_q

if __name__ == '__main__':

    batchsize = 8
    epsilon = 0.05
    discount = 0.95

    env = gym.make('CartPole-v1')
    env = bf.SubjectWrapper(env)
    buffer = bf.ReplayBuffer()
    env.attach_observer('replay_buffer', buffer)
    env.attach_observer('plotter', bf.Plot(blocksize=batchsize))

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    state_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_size = 12

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            input_size = state_size + n_actions
            self.q_net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, 1), nn.Softmax(dim=1))

        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            action = self.q_net(x)
            return action

    q_net = QNet()
    optim = torch.optim.Adam(q_net.parameters(), lr=1e-4)


    def policy(state):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            values, index = max_q(q_net, state, n_actions)
            return index.item()

    bf.episode(env, policy)

    done = True
    state = None
    for step in range(10000000):
        if done:
            state = env.reset()
            done = False
        else:
            action = policy(state)
            state, reward, done, info = env.step(action)

        deepq.train(buffer, q_net, optim, batchsize, n_actions, discount)
