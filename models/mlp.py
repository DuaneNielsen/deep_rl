import torch
from torch import nn
from torch.nn.functional import log_softmax
from torch.distributions import Categorical


class ValueHead(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.head = nn.Linear(hidden_dims, 1)

    def forward(self, state):
        return self.head(state)


class ActionHead(nn.Module):
    def __init__(self, hidden_dims, actions, exploration_noise):
        super().__init__()
        self.head = nn.Linear(hidden_dims, actions)
        self.actions = actions
        self.exploration_noise = exploration_noise

    def forward(self, state):
        action = log_softmax(self.head(state), dim=1)
        action = torch.log((1 - self.exploration_noise) * torch.exp(action) +
                           self.exploration_noise * torch.ones_like(action) / self.actions)
        return Categorical(logits=action)


class MLP(nn.Module):
    """
    policy(state) returns distribution over actions
    uses ScaledTanhTransformedGaussian as per Hafner
    """

    def __init__(self, in_features, hidden_dims, head):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dims), nn.SELU(inplace=True),
                                 nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                 head)

    def forward(self, state):
        return self.net(state)
