from torch import nn
from utills import init_weights, normalized_columns_initializer
import torch
import numpy as np
import torch.nn.functional as F


class HeadNet(nn.Module):
    def __init__(self, reshape_size, n_actions=4):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(reshape_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, h, w, num_channels):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(h=h, w=w, num_channels=num_channels)
        reshape_size = self.core_net.reshape_size
        self.net_list = nn.ModuleList([HeadNet(reshape_size=reshape_size, n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k=None):
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads


class CoreNet(nn.Module):
    def __init__(self, h, w, num_channels=4):
        super(CoreNet, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        self.reshape_size = convw * convh * 64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        x = x.view(-1, self.reshape_size)
        return x


