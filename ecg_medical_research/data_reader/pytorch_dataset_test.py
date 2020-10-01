import os
import pandas as pd
from ecg_medical_research.data_reader import pytorch_dataset
from matplotlib import pyplot as plt
import logging
from torch.utils.data import DataLoader


def test_dataset():
    ecg_dataset = pytorch_dataset.ECGToEchoDataset(
        excel_file='/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered_v2.xlsx',
        root_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_dicom',
        split='train')
    fig = plt.figure()

    for i in range(len(ecg_dataset)):
        sample = ecg_dataset[i]
        print(i, sample['ecg_signal'].shape, sample['echo'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.plot(sample['ecg_signal'][0][:500])
        if i == 3:
            plt.show()
            break


def test_dataset_with_transform():
    ecg_dataset = pytorch_dataset.ECGToEchoDataset(
        excel_file='/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered_v2.xlsx',
        root_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_dicom',
        transform=pytorch_dataset.ToTensor(),
    split='train')
    fig = plt.figure()

    for i in range(len(ecg_dataset)):
        sample = ecg_dataset[i]
        print(i, sample['ecg_signal'].shape, sample['echo'])
        print(type(sample['ecg_signal']), type(sample['echo']))
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.plot(sample['ecg_signal'].numpy()[0][:500])
        if i == 3:
            plt.show()
            break


def test_dataset_with_dataloader():
    ecg_dataset = pytorch_dataset.ECGToEchoDataset(
        excel_file='/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered_v2.xlsx',
        root_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_dicom',
        transform=pytorch_dataset.ToTensor(),
        split='validation')
    dataloader = DataLoader(ecg_dataset, batch_size=4,
                            shuffle=True, num_workers=4,
                            collate_fn=pytorch_dataset.collate_fn_pad)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['ecg_signal'].size(),
              sample_batched['echo'].size())
        print("mask: ", sample_batched['mask'])
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            plt.plot(sample_batched['ecg_signal'].numpy()[0][0][:500])
            plt.ioff()
            plt.show()
            break


def validate_dicom_file_existance():
    df = pd.read_excel('/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered_v2.xlsx')
    dicom_files = df['file name']
    not_found = 0
    found = 0
    for dicom_file in dicom_files:
        dicom_path = os.path.join('/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_dicom', f"r_{dicom_file}")
        if os.path.isfile(dicom_path):
            print(f"{dicom_file} found!")
            found += 1
        else:
            print(f"{dicom_file} not found!")
            not_found += 1
    print(f"{found} found.")
    print(f"{not_found} not found.")


def test_dataset_with_simetric_padding():
    ecg_dataset = pytorch_dataset.ECGToEchoDataset(
        excel_file='/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered_v2.xlsx',
        root_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset',
        transform=pytorch_dataset.ToTensor(),
        split='train')
    dataloader = DataLoader(ecg_dataset, batch_size=4,
                            shuffle=True, num_workers=1,
                            collate_fn=pytorch_dataset.collate_fn_simetric_padding)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['ecg_signal'].size(),
              sample_batched['echo'].size())
        # print("mask: ", sample_batched['mask'])
        # observe 4th batch and stop.
        if i_batch == 100:
            plt.figure()
            plt.plot(sample_batched['ecg_signal'].numpy()[0][0][:500])
            plt.ioff()
            plt.show()
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_dataset()
    # test_dataset_with_transform()
    # validate_dicom_file_existance()
    # test_dataset_with_dataloader()
    test_dataset_with_simetric_padding()