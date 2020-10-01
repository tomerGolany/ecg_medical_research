"""Temporal convolution classification model test."""
import torch
import numpy as np
from ecg_medical_research.architectures import tcn


if __name__ == "__main__":
    tcn_model = tcn.SimpleTCN(num_f_maps=64, num_classes=2)
    some_input = torch.from_numpy(np.random.normal(0, 1, (4, 12, 500))).float()
    out = tcn_model(some_input)