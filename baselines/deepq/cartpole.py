import torch
from torch import nn
import buffer as bf
import gym
import random
import algos.dqn as dqn

if __name__ == '__main__':

    batchsize = 8
    epsilon = 0.1
    discount = 0.9

    env = gym.make('CartPole-v1')
    env = bf.SubjectWrapper(env)
    buffer = bf.ReplayBuffer()
    env.attach_observer('replay_buffer', buffer)
    env.attach_observer('plotter', bf.Plot(blocksize=batchsize))

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    state_size = env.observation_space.shape[0]
    hidden_size = 12

    class QNet(nn.Module):
        def __init__(self, state_size, hidden_size, action_size):
            super().__init__()
            self.q_net = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, action_size))

        def forward(self, state):
            action = self.q_net(state)
            return action

    q_net = QNet(state_size=env.observation_space.shape[0], hidden_size=12, action_size=env.action_space.n)
    optim = torch.optim.SGD(q_net.parameters(), lr=5e-4)


    def policy(state):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(q_net(state), dim=1)
            return action.item()

    bf.episode(env, policy)

    batch_size = 8

    for step_n, (s, s_i, a, s_p, r, d, i) in enumerate(bf.transitions(env, policy, render=True)):
        if step_n < batch_size:
            continue
        if step_n > 20000:
            break
        dqn.train(buffer, q_net, optim, batch_size, discount=discount)
