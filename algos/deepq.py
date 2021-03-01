import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import algos.utils
from torch.nn.functional import one_hot


def replicate_states(state, n_actions):
    """
    Generates a copy of each state for each action
    state: N x S dimensions (also with  N, H, W, or N, ...)
    n_actions: number of actions agent can chose from
    ```
    replicate_states(torch.rand(2, 3), 3)
    tensor([[0.9497, 0.9230, 0.4086],
            [0.9497, 0.9230, 0.4086],
            [0.9497, 0.9230, 0.4086],
            [0.9554, 0.2950, 0.3355],
            [0.9554, 0.2950, 0.3355],
            [0.9554, 0.2950, 0.3355]])
    ```
    """

    return state.repeat_interleave(n_actions, 0)


def replicate_actions(N, n_actions):
    """
    Generates N repeats of each actions

    ```
    replicate_actions(N=2, n_actions=3)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    ```
    """
    return torch.eye(n_actions).repeat(N, 1)


def max_q(q_net, state, n_actions):
    """
    Computes the best action for the given state
    q_net: value function q(state, action)
    state: N, ... states
    n_actions: number of actions in environment
    returns: (value, action)
        value: N, 1 values
        action: N, 1 actions
    """
    N = state.shape[0]
    states = replicate_states(state, n_actions)
    actions = replicate_actions(N, n_actions)
    values = q_net(states, actions)
    values = values.reshape(N, n_actions)
    return values.max(dim=1, keepdim=True)


def train(buffer, q_net, optim, batch_size, n_actions, discount, device='cpu', dtype=torch.float):

    sampler = SubsetRandomSampler(random.sample(range(len(buffer)), batch_size))

    dl = DataLoader(buffer, batch_size=batch_size, sampler=sampler)

    for s, i, a, s_p, r, d, i_p in dl:
        s, s_p, r, d = algos.utils.to(s, s_p, r, d, device=device, dtype=dtype)
        a = one_hot(a, n_actions).type(dtype).to(device)

        optim.zero_grad()
        v0 = q_net(s, a)
        with torch.no_grad():
            v1, _ = max_q(q_net, s_p, n_actions)
            v1[d] = 0.0
            v1 = r + v1 * discount
        loss = torch.mean((v1 - v0) ** 2)

        loss.backward()
        optim.step()


