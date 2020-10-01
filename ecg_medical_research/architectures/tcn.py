"""Temporal convolution classification model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


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
    def __init__(self, num_f_maps, num_classes):
        super(SimpleTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(12, num_f_maps, 1)
        self.l1 = DilatedResidualLayer(1, num_f_maps, num_f_maps)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = self.l1(out, mask)
        out = self.conv_out(out)  # [N, 2, 5499]
        out = F.avg_pool1d(out, out.size()[2])
        out = torch.squeeze(out, dim=2)
        return out