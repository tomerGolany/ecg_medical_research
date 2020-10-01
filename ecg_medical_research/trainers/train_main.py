"""Main file to run training and evaluation."""
import os
from ecg_medical_research.trainers import train_lib
import torch
import logging
from ecg_medical_research.data_reader.dataset import ecg_to_echo_dataset
import pandas as pd
from ecg_medical_research.evaluation import quality
import numpy as np


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Training with device %s...", device)
    # model_dir = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/model_outputs'
    model_dir = '.'
    model_dir = os.path.join(model_dir, 'resnet50_rms_prop_4')
    train_lib.train(
        # excel_file='../data_reader/dataset/dataset_full_details.csv',
        excel_file='../data_reader/dataset/full_dataset_with_see_below_2.csv',
        # dicom_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset',
        dicom_dir='/home/tomer.golany/dataset',
        model_dir=model_dir,
        batch_size=64,
        num_iterations=20000,
        device=device)


if __name__ == "__main__":
    main()
