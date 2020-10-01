"""Simple convolution network."""

import torch
from torch import nn
import torch.nn.functional as F


class Lenet(nn.Module):
    def __init__(self, input_features_size: int):
        super().__init__()
        # Input shape: [batch_size, 12, L]

        # -> [batch_size, 32, L]
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, padding=2)
        # -> [batch_size, 32, L/2]
        self.pool1 = nn.MaxPool1d(2, 2)
        # -> [batch_size, 64, L/2]
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # -> [batch_size, 64, L/4]
        self.pool2 = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=87936, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=80)
        self.fc3 = nn.Linear(in_features=80, out_features=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x