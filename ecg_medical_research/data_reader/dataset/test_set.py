"""Create Pytorch dataset which holds all ECGs from the Test-Set."""
import torch
from torch.utils.data import Dataset
from ecg_medical_research.data_reader.dataset import metadata
from ecg_medical_research.data_reader.dataset import ecg_to_echo_dataset
from ecg_medical_research.evaluation import quality
import pandas as pd
import os
import logging
from ecg_medical_research.data_reader import patient


class TestSet(Dataset):
    def __init__(self, transform=None, threshold_35=False):
        self.threshold_35 = threshold_35
        # self.annotations_df = pd.read_csv(metadata.EXCEL_DATASET_FILE)
        self.dicom_files = os.listdir(metadata.DICOM_DIR)
        # self.annotations_df = ecg_to_echo_dataset.validate_dicom_and_excel(self.annotations_df, self.dicom_files)
        # logging.info("Number of annotations after matching with dicom files: %d", len(self.annotations_df))

        #
        # Keep only test annotations and dicom files:
        #
        test_set_df = pd.read_excel(metadata.TEST_SET_FILE)

        #
        # Remove blank ECGs:
        #
        test_set_df['ecg_number'] = test_set_df.index.map(lambda x: x + 1)
        self.test_set_df = test_set_df[~test_set_df['ecg_number'].isin(quality.ECGS_TO_IGNORE)]

        perfect_ecg_numbers = quality.test_set_quality_keep(quality.EcgQuality.PERFECT)
        test_set_df_only_perfect = test_set_df[test_set_df['ecg_number'].isin(perfect_ecg_numbers)]

        all_except_artifacts = quality.test_set_quality_filter(quality.EcgQuality.SEVERE_ARTIFACTS)
        test_set_df_without_artifacts = test_set_df[test_set_df['ecg_number'].isin(all_except_artifacts)]

        logging.info("Total Number of ECGs in the test-set: %d", len(test_set_df))
        logging.info("Number of Perfect ECGs: %d", len(test_set_df_only_perfect))
        logging.info("Number of ECGs without Artifacts: %d", len(test_set_df_without_artifacts))

        # self.test_set_files = test_set_df['file name']
        # self.ecg_numbers = test_set_df['ecg_number']
        # self.test_set_df_without_artifacts = test_set_df_without_artifacts['file name']
        # self.test_set_df_only_perfect = test_set_df_only_perfect['file name']

        # self.annotations_df = self.annotations_df[self.annotations_df['file name'].isin(self.test_set_files)]
        self.transform = transform
        logging.info("Final number of annotations: %d", len(self.test_set_df))

    def __len__(self):
        # return len(self.annotations_df)
        return len(self.test_set_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dicom_path = os.path.join(metadata.DICOM_DIR, f"{self.test_set_df['file name'].iloc[idx]}")
        patient_obj = patient.Patient(patient_dicom_path=dicom_path)
        ecg_number = self.test_set_df['ecg_number'].iloc[idx]
        age = patient_obj.age
        number_of_samples = patient_obj.number_of_samples
        sampling_rate = patient_obj.sampling_frequency
        gender = patient_obj.gender
        duration = patient_obj.decoded_dcom_obj.duration
        if not self.threshold_35:
            echo_result = self.test_set_df['label'].iloc[idx]
        else:
            label_str = self.test_set_df['cognos_reserved_text13'].iloc[idx]
            if label_str in metadata.HEALTHY_35:
                echo_result = 1
            else:
                if label_str not in metadata.SICK_35:
                    raise AssertionError(f"{label_str}")
                echo_result = 0
        sample = {# 'ecg_signal_unfiltered': patient_obj.unfiltered_signals,
                  # 'ecg_signal_filtered': patient_obj.filtered_signals,
                  'ecg_signal_filtered': patient_obj.sub_sampled_ecg,
                  'echo': echo_result,
                  'ecg_number': ecg_number,
                  'dicom_file': self.test_set_df['file name'].iloc[idx],
                  'age': age,
                  'number_of_samples': number_of_samples,
                  'sampling_rate': sampling_rate,
                  'gender': gender,
                  'duration': duration}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":

    # dicom_files = os.listdir(metadata.DICOM_DIR)
    # test_set_df = pd.read_excel(metadata.TEST_SET_FILE)
    # test_set_df['ecg_number'] = test_set_df.index.map(lambda x: x + 1)
    # # test_set_df = test_set_df[~test_set_df['ecg_number'].isin(quality.ECGS_TO_IGNORE)]
    # dicom_blank = test_set_df[test_set_df['ecg_number'] == 285]['file name'].iloc[0]
    # print(dicom_blank)
    # dicom_path = os.path.join(metadata.DICOM_DIR, dicom_blank)
    # patient_obj = patient.Patient(patient_dicom_path=dicom_path)
    # print(patient_obj)
    # filtered = patient_obj.filtered_signals
    # from matplotlib import pyplot as plt
    # for lead in filtered:
    #     print("Max: ", max(lead))
    #     print("Min: ", min(lead))
    #     # for x in list(reversed(lead))[:100]:
    #     #     print(x)
    #     # print(lead)
    #     plt.figure()
    #     plt.plot(lead)
    #     plt.show()
    #
    # import numpy as np
    # from scipy import signal
    # bad_ecgs = []
    # for i, row in test_set_df.iterrows():
    #     dicom_path = os.path.join(metadata.DICOM_DIR, row['file name'])
    #     patient_obj = patient.Patient(patient_dicom_path=dicom_path)
    #     filtered = patient_obj.filtered_signals
    #     for lead in filtered:
    #         # lead = signal.medfilt(lead, kernel_size=9)
    #         if (np.all(lead[1000:2000] == lead[1000])):
    #             bad_ecgs.append(row['ecg_number'])
    #             break
    #     # print("Max: ", max(lead))
    #     # print("Min: ", min(lead))
    #     # # for x in list(reversed(lead))[:100]:
    #     # #     print(x)
    #     # # print(lead)
    #     # plt.figure()
    #     # plt.plot(lead)
    #     # plt.show()
    # print(bad_ecgs)
    ages = []
    durations = []
    num_samples = []
    samp_rates = []
    genders = []
    ids = []
    test_set = TestSet()
    for data in test_set:
        patient_obj = data['patient_obj']
        age = patient_obj.age
        number_of_samples = patient_obj.number_of_samples
        sampling_rate = patient_obj.sampling_frequency
        gender = patient_obj.gender
        duration = patient_obj.decoded_dcom_obj.duration

        ages.append(age)
        num_samples.append(number_of_samples)
        samp_rates.append(sampling_rate)
        genders.append(gender)
        durations.append(duration)
        ids.append(patient_obj.id)

    df = pd.DataFrame(list(zip(ids, ages, num_samples, samp_rates, genders, durations)),
                      columns=['id', 'age', 'num_samples',
                               'samp_rates', 'gender',
                               'duration'])
    print(df.head())
    df.to_csv('test_data_info.csv')