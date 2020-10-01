from ecg_medical_research.architectures import lenet
import torch
import numpy as np


def simple_input():
    net = lenet.Lenet(input_features_size=80)
    fake_input = torch.from_numpy(np.random.normal(size=[2, 12, 80])).float()
    print("Fake input shape: ", fake_input.size())
    out = net(fake_input)
    print("output shape: ", out.size())


if __name__ == "__main__":
    simple_input()