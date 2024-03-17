import torch
from torch import nn
from torch.nn.functional import log_softmax
from torch.distributions import Categorical


class ValueHead(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.head = nn.Linear(hidden_dims, 1, bias=False)

    def forward(self, state):
        return self.head(state)


class ActionHead(nn.Module):
    def __init__(self, hidden_dims, actions):
        super().__init__()
        self.head = nn.Linear(hidden_dims, actions, bias=False)

    def forward(self, state, exploration_noise=0.):
        action_probs = torch.softmax(self.head(state), dim=-1)
        uniform_kernel = torch.ones_like(action_probs) / action_probs.size(-1)
        action = (1 - exploration_noise) * action_probs + exploration_noise * uniform_kernel
        return Categorical(probs=action)


class MLP(nn.Module):
    """
    policy(state) returns distribution over actions
    uses ScaledTanhTransformedGaussian as per Hafner
    """

    def __init__(self, in_features, hidden_dims, head):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dims), nn.Hardswish(inplace=True),
                                 nn.Linear(hidden_dims, hidden_dims), nn.Hardswish(inplace=True),
                                 head)

    def forward(self, state, *args, **kwargs):
        return self.net(state, *args, **kwargs)
