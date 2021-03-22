import torch
from torch import nn
from torch.nn.functional import max_pool2d, log_softmax
from torch.distributions import Categorical


class A2CNet(nn.Module):
    """
    policy(state) returns distribution over actions
    uses ScaledTanhTransformedGaussian as per Hafner
    """
    def __init__(self, hidden_dims, actions, exploration_noise):
        super().__init__()
        self.actions = actions
        self.exploration_noise = exploration_noise
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.value = nn.Sequential(nn.Linear(512, hidden_dims), nn.SELU(inplace=True),
                                 nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                 nn.Linear(hidden_dims, 1))

        self.action = nn.Sequential(nn.Linear(512, hidden_dims), nn.SELU(inplace=True),
                                 nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                 nn.Linear(hidden_dims, actions))

    def forward(self, state):
        l1 = self.conv1(state.permute(0, 3, 1, 2))
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(l5)

        value = self.value(l6.flatten(start_dim=1))
        action = log_softmax(self.action(l6.flatten(start_dim=1)), dim=1)
        action = torch.log((1 - self.exploration_noise) * torch.exp(action) +
                           self.exploration_noise * torch.ones_like(action)/self.actions)
        a_dist = Categorical(logits=action)
        return value, a_dist


class SmallNet(nn.Module):
    """
    policy(state) returns distribution over actions
    uses ScaledTanhTransformedGaussian as per Hafner
    """
    def __init__(self, hidden_dims, actions, exploration_noise):
        super().__init__()
        self.actions = actions
        self.exploration_noise = exploration_noise
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=8, stride=4, padding=0),
            nn.SELU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.SELU(inplace=True))

        self.value = nn.Sequential(nn.Linear(32 * 9 * 9, 256), nn.SELU(inplace=True),
                                 nn.Linear(256, 1))

        self.action = nn.Sequential(nn.Linear(32 * 9 * 9, 256), nn.SELU(inplace=True),
                                 nn.Linear(256, actions))

    def forward(self, state):
        l1 = self.conv1(state.permute(0, 3, 1, 2))
        l2 = self.conv2(l1)

        value = self.value(l2.flatten(start_dim=1))
        action = log_softmax(self.action(l2.flatten(start_dim=1)), dim=1)
        action = torch.log((1 - self.exploration_noise) * torch.exp(action) +
                           self.exploration_noise * torch.ones_like(action)/self.actions)
        a_dist = Categorical(logits=action)
        return value, a_dist