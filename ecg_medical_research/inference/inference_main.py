"""Run inference on a trained model."""
from ecg_medical_research.evaluation import export_model
from ecg_medical_research.data_reader.dataset import test_set, ecg_to_echo_dataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import torch
import pandas as pd
from typing import Optional


def predict_on_dataset(net_object, dataloader_object, device):
    """Predict on all samples in a dataloader."""
    logging.info("Starting predictions on %d ECG records...", len(dataloader_object.dataset))
    predictions = np.array([]).reshape((0, 2))
    ground_truths = np.array([]).reshape((0, 2))
    dicom_names = []
    ecg_numbers = []
    ages = []
    durations = []
    num_samples = []
    samp_rates = []
    genders = []
    number_of_ecg_predicted = 0
    with torch.no_grad():
        # net_object.eval()
        for i_data, data in enumerate(dataloader_object):
            ecg_batch, labels_batch, ecg_numbers_batch = (
                data['ecg_signal_filtered'].to(device), data['echo'].to(device), data['ecg_number'].to(device))
            dicom_names += data['dicom_file']
            ages += data['age']
            num_samples += data['number_of_samples']
            samp_rates += data['sampling_rate']
            genders += data['gender']
            durations += data['duration']

            outputs = net_object(ecg_batch)  # Shape: [batch size, 2]
            _, predicted = torch.max(outputs.data, dim=1)
            ecg_numbers += list(ecg_numbers_batch.cpu().detach().numpy())
            number_of_ecg_predicted += labels_batch.size(0)
            if (i_data + 1) % 5 == 0:
                logging.info("Predicted %d/%d...", number_of_ecg_predicted, len(dataloader_object.dataset))

            labels_one_hot = torch.nn.functional.one_hot(labels_batch, 2)  # 1 = Normal Echo
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            ground_truths = np.concatenate((ground_truths, labels_one_hot.cpu().detach().numpy()))
            predictions = np.concatenate((predictions, probabilities.cpu().detach().numpy()))
        logging.info("Completed predictions on %d ECG records.", number_of_ecg_predicted)
        # net_object.train()
        df = pd.DataFrame(list(zip(ecg_numbers, ground_truths[:, 1], predictions[:, 1], dicom_names,
                                   ages, num_samples, samp_rates, genders, durations)),
                          columns=['ecg_number', 'ground_truth', 'prediction', 'dicom file',
                                   'age', 'num_samples',
                                   'samp_rates', 'gender',
                                   'duration'
                                   ])
        df['label'] = df['ground_truth'].apply(lambda gt: 'normal' if gt == 1 else 'sick')
        df.to_csv("inference.csv", index=False)
        return df


def run_inference(checkpoint_path: Optional[str] = None, network=None, device="cpu"):

    if network is None:
        net = export_model.load_model_from_checkpoint(checkpoint_path, device)
    else:
        net = network
    all_test_ecg_dataset = test_set.TestSet(transform=ecg_to_echo_dataset.ToTensor(), threshold_35=False)
    batch_size = 50

    all_test_loader = DataLoader(all_test_ecg_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                                 # collate_fn=ecg_to_echo_dataset.collate_fn_simetric_padding)
    logging.info("Running inference on All Test ECGs...")
    inference_df = predict_on_dataset(net, all_test_loader, device)
    return inference_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # run_inference("/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/trainers/resnet50_rms_prop_4/checkpoints/checkpoint_epoch_41_iters_9600")
    run_inference("/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/trainers/resnet50_rms_prop_5/checkpoints/checkpoint_epoch_1_iters_100")