"""Export a trained model from a checkpoint."""
import torch
from ecg_medical_research.architectures import resnet, mayo
import logging


def load_model_from_checkpoint(checkpoint_path, device):
    """Init a model and load it weights from a trained checkpoint."""
    net = resnet.resnet50().to(device)
    # net = mayo.MayoNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['resnet'])
    logging.info("Successfully loaded model from checkpoint %s", checkpoint_path)
    return net


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = "/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/trainers/resnet50_rms_prop_4/checkpoints/checkpoint_epoch_41_iters_9600"
    device = "cpu"
    net = load_model_from_checkpoint(checkpoint_path, device)
