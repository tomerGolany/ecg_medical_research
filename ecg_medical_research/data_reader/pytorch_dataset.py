"""Parse the excel file and create a pytorch dataset."""
import os
import numpy as np
import io
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from ecg_medical_research.data_reader import patient
import logging
import torch.nn.functional as F


def collate_fn_pad(batch):
    """Pads batch of variable length.

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    ecg_signals = [b['ecg_signal'].permute(1, 0) for b in batch]
    lengths = [e['ecg_signal'].size()[1] for e in batch]
    max_length = max(lengths)
    ecg_signals_padded = torch.nn.utils.rnn.pad_sequence(ecg_signals, batch_first=True)
    labels = [b['echo'] for b in batch]
    ecg_signals_padded = ecg_signals_padded.permute(0, 2, 1)
    # Compute mask:
    mask = torch.ones(len(batch), max_length, dtype=torch.float)
    for i, l in enumerate(lengths):
        mask[i, l:] = torch.zeros(max_length - l, dtype=torch.float)
    mask = mask.unsqueeze(1)
    return {'ecg_signal': ecg_signals_padded, 'echo': torch.stack(labels), 'mask': mask}


def collate_fn_simetric_padding(batch):
    """Pad element symmetrically.

    :param batch: List of samples from ECGToEchoDataset
    :return: matrix of the values.
    """
    ecg_signals = [sample['ecg_signal'] for sample in batch]  # Shape of each signal: [12, L]
    lengths = [e.size()[1] for e in ecg_signals]
    max_length = max(lengths)
    # max_length = 10000
    padded_signals = []
    for ecg_signal, length in zip(ecg_signals, lengths):
        if length < max_length:
            if (max_length - length) % 2 == 0:
                pad_size = ((max_length - length) // 2, (max_length - length) // 2)
            else:
                pad_size = ((max_length - length) // 2, ((max_length - length) // 2) + 1)
            padded_signals.append(F.pad(ecg_signal, pad_size, "constant", 0))
        else:
            padded_signals.append(ecg_signal)
    labels = [b['echo'] for b in batch]
    return {'ecg_signal': torch.stack(padded_signals), 'echo': torch.stack(labels)}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {k: (torch.from_numpy(v) if type(v) == np.ndarray else torch.tensor(v)) for k, v in sample.items()}


class ECGToEchoDataset(Dataset):
    """Maps 12-Lead ecg to normal/ubnormal echo exam."""

    def __init__(self, excel_file: str, root_dir: str, split: str, transform=None):
        """Initialize the ECG-Echo dataset.

        Args:
            excel_file: Path to the excel file with annotations.
            root_dir: Directory with all the dicom files.
            split: Train/Val/Test.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ecg_echo_df = pd.read_excel(excel_file)
        self.root_dir = root_dir
        self.transform = transform
        # Filter out .dcm files that don't exists:
        list_of_existing_dcm = os.listdir(self.root_dir)
        list_of_existing_dcm = [f for f in list_of_existing_dcm if f.endswith('.dcm')]
        existing_files_df = self.ecg_echo_df[self.ecg_echo_df['file name'].isin(list_of_existing_dcm)]
        non_existing_files = self.ecg_echo_df[~self.ecg_echo_df['file name'].isin(list_of_existing_dcm)]
        print(non_existing_files)

        logging.info(f"Total number of .dcm files in excel file: {len(self.ecg_echo_df)}")

        total_number_of_samples = len(existing_files_df)
        logging.info(f"Number of matched .dcm files: {len(existing_files_df)}")
        # Slice to desired split:
        if split == 'train':
            self.ecg_echo_df = existing_files_df[:int(0.8 * total_number_of_samples)]
        elif split == 'validation':
            self.ecg_echo_df = existing_files_df[int(0.8 * total_number_of_samples):]
        logging.info(f"Number of samples in split {split}: {len(self.ecg_echo_df)}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.ecg_echo_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dicom_path = os.path.join(self.root_dir, f"{self.ecg_echo_df['file name'].iloc[idx]}")
        patient_obj = patient.Patient(patient_dicom_path=dicom_path)
        echo_result = self.ecg_echo_df['label'].iloc[idx]
        sample = {'ecg_signal': patient_obj.unfiltered_signals, 'echo': echo_result}

        if self.transform:
            sample = self.transform(sample)
        return sample
