"""Temporal convolution classification model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class TemporalBlock(nn.Module):  # It's not clear if they used residual connections or not.
    def __init__(self, in_channels, num_filters, kernel_size, dilation_rate=0, padding=0):
        super(TemporalBlock, self).__init__()
        # [b, c, h, w] = [b, 1, 12, 1000]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=[1, kernel_size],
                               padding=[0, padding], dilation=[0, dilation_rate])  # [b, num_filters, 12, 1000]
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=[1, 1])

    def forward(self, x):
        # input_x [b, 1, 12, 1000]
        x = self.conv1(x)  # [b, 16, ]
        x = F.relu(x)
        x = self.conv2(x)
        return x


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, temporal_size=3):
        """Dilated residual layer.

        :param dilation: dilation rate.
        :param in_channels: Input channel of the input to the layer.
        :param out_channels: Number of filters.
        :param temporal_size: Temporal filter size.
        """
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, temporal_size, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        # out = self.dropout(out)
        return (x + out) * mask


class SimpleTCN(nn.Module):
    def __init__(self, num_f_maps, num_classes, num_layers):
        super(SimpleTCN, self).__init__()
        # input_x [b, 1, 12, 1000]
        self.conv_1x1 = nn.Conv2d(1, num_f_maps, 1)
        self.dilated_residual_layers = []
        for i in range(num_layers):
            self.dilated_residual_layers.append(TemporalBlock(in_channels=num_f_maps, num_filters=num_f_maps,
                                                              kernel_size=3,
                                                              dilation_rate=2 ** i,
                                                              padding=2 ** i))


        self.conv_out = nn.Conv2d(in_channels=num_f_maps, out_channels=num_filters, kernel_size=[1, 1])

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = self.l1(out)
        out = self.conv_out(out)
        return out


if __name__ == "__main__":
    fake_ecg = torch.ones((2, 1, 12, 1000))
    conv_1x1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
    out = conv_1x1(fake_ecg)
    print(out.shape)
    conv_dilated = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 3],
                               padding=[0, 1], dilation=[1, 1])
    out = conv_dilated(out)
    print(out.shape)
    out = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1])(out)
    print(out.shape)