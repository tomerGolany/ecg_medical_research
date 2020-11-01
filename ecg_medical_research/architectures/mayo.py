"""Implemetation of a Convoluition network based on paper by Mayo clinic.

Paper title: Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram
Link: https://www.nature.com/articles/s41591-018-0240-2

architecture description:
https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-018-0240-2/MediaObjects/41591_2018_240_MOESM1_ESM.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):  # It's not clear if they used residual connections or not.
    def __init__(self, in_channels, num_filters, kernel_size, max_pooling_factor, padding=0):
        super(TemporalBlock, self).__init__()
        # [b, c, h, w] = [b, 1, 12, 1000]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=[1, kernel_size],
                               padding=[0, padding])  # [b, num_filters, 12, 1000]
        self.bn = nn.BatchNorm2d(num_features=num_filters)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d([1, max_pooling_factor])

    def forward(self, x):
        # input_x = x  # [b, 12, 1000]
        x = self.conv1(x)  # [b, 16, ]
        x = self.bn(x)
        x = F.relu(x)
        x = self.max_pool(x)
        return x


class SpatialBlock(nn.Module):
    def __init__(self):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[12, 1],
                               padding=0)  # [b, K, T]
        self.bn = nn.BatchNorm2d(num_features=128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class MayoNet(nn.Module):
    def __init__(self):
        super(MayoNet, self).__init__()
        self.temporal_conv1 = TemporalBlock(in_channels=1, num_filters=16, kernel_size=5,
                                            max_pooling_factor=2, padding=14)
        self.temporal_conv2 = TemporalBlock(in_channels=16, num_filters=16, kernel_size=5,
                                            max_pooling_factor=2, padding=2)
        self.temporal_conv3 = TemporalBlock(in_channels=16, num_filters=32, kernel_size=5,
                                            max_pooling_factor=4, padding=2)
        self.temporal_conv4 = TemporalBlock(in_channels=32, num_filters=32, kernel_size=3,
                                            max_pooling_factor=2, padding=1)
        self.temporal_conv5 = TemporalBlock(in_channels=32, num_filters=64, kernel_size=3,
                                            max_pooling_factor=2, padding=1)
        self.temporal_conv6 = TemporalBlock(in_channels=64, num_filters=64, kernel_size=3,
                                            max_pooling_factor=4, padding=1)

        self.spatial_conv = SpatialBlock()

        self.fc_1 = nn.Linear(in_features=19 * 128, out_features=64)
        self.bn_1 = nn.BatchNorm1d(64)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=64, out_features=32)
        self.bn_2 = nn.BatchNorm1d(32)
        self.relu_2 = nn.ReLU()
        self.prediction_layer = nn.Linear(in_features=32, out_features=2)
        # self.drop = nn.Dropout()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal_conv1(x)

        x = self.temporal_conv2(x)

        x = self.temporal_conv3(x)

        x = self.temporal_conv4(x)

        x = self.temporal_conv5(x)

        x = self.temporal_conv6(x)

        x = self.spatial_conv(x)
        # print(x.shape)
        # x = torch.squeeze(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.prediction_layer(x)

        return x


if __name__ == "__main__":
    # print(torch.__version__)
    # temporal_block = TemporalBlock(in_channels=1, num_filters=16, kernel_size=5, max_pooling_factor=2)
    # print(temporal_block.conv1.weight.shape)
    # fake_ecg = torch.ones((2, 12, 1, 1000))
    # fake_ecg = fake_ecg.view(2 * 12, 1, 1000)
    # out = temporal_block(fake_ecg)
    # print(out.shape)
    # print("MAYO")
    # mayo_net = MayoNet()
    # fake_ecg = torch.ones((2, 12, 1000, 1))
    # fake_ecg = fake_ecg.view(2 * 12, 1, 1000)
    #
    # out = mayo_net(fake_ecg)
    # print(out.shape)

    fake_ecg = torch.ones((32, 12, 5000))
    mayo_net = MayoNet()
    temporal_block = TemporalBlock(in_channels=1, num_filters=16, kernel_size=5, max_pooling_factor=2, padding=14)
    out = mayo_net(fake_ecg)
    # print(temporal_block.weight.shape)
    print(out.shape)

    # import tensorflow as tf

    # print(tf.__version__)
    # input_shape = (2, 12, 1000, 1)
    # x = tf.random.normal(input_shape)
    # conv = tf.keras.layers.Conv1D(
    #     64, 5, activation='relu', input_shape=input_shape[2:])
    # y = conv(x)
    # print(conv.kernel)
    # print(y.shape)