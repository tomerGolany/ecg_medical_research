import torch
import pandas as pd
import os
from ecg_medical_research.data_reader import patient
import numpy as np
from torch.utils.data import Dataset
from ecg_medical_research.data_reader import parse_ecg_echo_excel
from ecg_medical_research.data_reader.dataset import metadata
from ecg_medical_research.evaluation import quality
import enum
import logging
from torch.utils.data import DataLoader


EXCEL_DATASET_FILE = '../data_reader/dataset/full_dataset_with_see_below_2.csv'

DICOM_DIR = '/home/tomer.golany/dataset'
DICOM_DIR = "/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset"

BLANK_ECGS_CSV = "/home/tomer.golany/ecg_medical_research/ecg_medical_research/data_reader/dataset/blank_ecgs.csv"
BLANK_ECGS_CSV = "/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/blank_ecgs.csv"

class TestType(enum.Enum):
    ALL_TEST = 1
    ONLY_PERFECT = 2
    WITHOUT_ARTIFACTS = 3


def collate_fn_simetric_padding(batch):
    """Pad element symmetrically.

    :param batch: List of samples from ECGToEchoDataset
    :return: matrix of the values.
    """
    ecg_signals = [sample['ecg_signal_filtered'] for sample in batch]  # Shape of each signal: [12, L]
    lengths = [e.size()[1] for e in ecg_signals]
    # max_length = max(lengths)
    max_length = 10000
    padded_signals = []
    for ecg_signal, length in zip(ecg_signals, lengths):
        if length < max_length:
            if (max_length - length) % 2 == 0:
                pad_size = ((max_length - length) // 2, (max_length - length) // 2)
            else:
                pad_size = ((max_length - length) // 2, ((max_length - length) // 2) + 1)
            padded_signals.append(torch.nn.functional.pad(ecg_signal, pad_size, "constant", 0))
        else:
            padded_signals.append(ecg_signal)
    labels = [b['echo'] for b in batch]
    ages = [b['age'] for b in batch]
    number_of_samples = [b['number_of_samples'] for b in batch]
    sampling_rate = [b['sampling_rate'] for b in batch]
    gender = [b['gender'] for b in batch]
    duration = [b['duration'] for b in batch]

    dicom_files = [b['dicom_file'] for b in batch]
    if 'ecg_number' in batch[0]:
        ecg_numbers = [b['ecg_number'] for b in batch]
        return {'ecg_signal_filtered': torch.stack(padded_signals), 'echo': torch.stack(labels),
            'ecg_number': torch.stack(ecg_numbers), "dicom_file": dicom_files, 'age': ages,
                'number_of_samples': number_of_samples, 'sampling_rate': sampling_rate,
                'gender': gender, 'duration': duration}
    else:
        return {'ecg_signal_filtered': torch.stack(padded_signals), 'echo': torch.stack(labels),
                "dicom_file": dicom_files, 'age': ages,
                'number_of_samples': number_of_samples, 'sampling_rate': sampling_rate,
                'gender': gender, 'duration': duration}

def fill_see_below(full_dataset_csv, see_below_excel):
    full_df = pd.read_csv(full_dataset_csv)
    see_below_df = pd.read_excel(see_below_excel)
    num_see_below = 0
    i = 0
    for r_i, row in full_df.iterrows():
        if row['label'] == -1:
            num_see_below += 1
            dcm_name = row['file name']
            label_row = see_below_df[see_below_df['file name'] == dcm_name]
            if len(label_row) > 1:
                print("More than one row for same dcm.")
                raise ValueError()
            elif len(label_row) == 0:
                print("Dcm not found in see below.")
                raise ValueError()
            else:
                i += 1
                label = parse_ecg_echo_excel.get_label(label_row.iloc[0])
                print("label: ", label)
                full_df.at[r_i, 'label'] = label
                label_str = label_row.iloc[0]['cognos_reserved_text13']
                print(label_str)
                full_df.at[r_i, 'cognos_reserved_text13'] = label_str
    print("Num see below ", num_see_below)
    print("num translations: ", i)
    full_df.to_csv('full_dataset_with_see_below_2.csv', index=False)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        d = {}
        for k, v in sample.items():
            if type(v) == np.ndarray:
                d[k] = torch.from_numpy(v).float()
            elif type(v) == str:
                d[k] = v
            elif type(v) == patient.Patient:
                d[k] = v
            else:
                d[k] = torch.tensor(v)
        return d


def validate_dicom_and_excel(annotations_df, dicom_files):
    logging.info("Parsing Excel dataset file...")
    logging.info("Number of lines: %d", len(annotations_df))
    logging.info("Number of unique dicom files in excel: %d", len(annotations_df['file name'].unique()))
    logging.info("Reading dicom files...")
    logging.info("Number of dicom files: %d", len(dicom_files))
    logging.info("Number of dicom files that exist in the excel file: %d",
                 len(annotations_df[annotations_df['file name'].isin(dicom_files)]))
    return annotations_df[annotations_df['file name'].isin(dicom_files)]


def parse_num_samples(row, dicom_dir):
    dicom_path = os.path.join(dicom_dir, f"{row['file name']}")
    patient_obj = patient.Patient(patient_dicom_path=dicom_path)
    return patient_obj.filtered_signals.shape[1]


def parse_sampling_rate(row, dicom_dir):
    dicom_path = os.path.join(dicom_dir, f"{row['file name']}")
    patient_obj = patient.Patient(patient_dicom_path=dicom_path)
    return patient_obj.sampling_frequency


class ECGToEchoDataset(Dataset):
    def __init__(self, excel_path, dicom_dir, split_name=None, transform=None, threshold_35=False,
                 test_split_type=TestType.ALL_TEST):
        self.threshold_35 = threshold_35
        self.excel_path = excel_path
        self.dicom_dir = dicom_dir
        self.annotations_df = pd.read_csv(excel_path)
        self.dicom_files = os.listdir(dicom_dir)
        self.annotations_df = validate_dicom_and_excel(self.annotations_df, self.dicom_files)
        self.test_split_type = test_split_type
        # self.filter_samples()
        # self.filter_see_below()
        print("After filtering: ", len(self.annotations_df))
        # Filter test set:
        # test_set_path = '../data_reader/excel_files/test_set_v2.xlsx'
        test_set_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/test_set_v2.xlsx'
        test_set_df = pd.read_excel(test_set_path)

        test_set_df['ecg_number'] = test_set_df.index.map(lambda x: x + 1)
        test_set_df = test_set_df[~test_set_df['ecg_number'].isin(quality.ECGS_TO_IGNORE)]
        perfect_ecg_numbers = quality.test_set_quality_keep(quality.EcgQuality.PERFECT)
        test_set_df_only_perfect = test_set_df[test_set_df['ecg_number'].isin(perfect_ecg_numbers)]
        all_except_artifacts = quality.test_set_quality_filter(quality.EcgQuality.SEVERE_ARTIFACTS)
        test_set_df_without_artifacts = test_set_df[test_set_df['ecg_number'].isin(all_except_artifacts)]
        print(f'All test set: {len(test_set_df)}')
        print(f'Only perfect ECGS: {len(test_set_df_only_perfect)}')
        print(f'Without Artifacts ECGS: {len(test_set_df_without_artifacts)}')

        self.test_set_files = test_set_df['file name']
        self.test_set_df_without_artifacts = test_set_df_without_artifacts['file name']
        self.test_set_df_only_perfect = test_set_df_only_perfect['file name']

        if split_name != 'test':
            self.filter_test_set()
        print("After filtering test set:", len(self.annotations_df))
        if split_name is not None:
            self.annotations_df = self.split(name=split_name)
        self.transform = transform
        print("Final length: ", len(self.annotations_df))

        print("Filtering blank ECGS:")
        self.filter_blank_ecgs()
        print("Length after filtering: ", len(self.annotations_df))

    def filter_blank_ecgs(self):
        blank_ecgs_df = pd.read_csv(BLANK_ECGS_CSV)
        self.annotations_df = self.annotations_df[~self.annotations_df['file name'].isin(blank_ecgs_df['dicom_file'])]

    def filter_samples(self):
        self.annotations_df = self.annotations_df[self.annotations_df.num_samples == 5499]

    def filter_see_below(self):
        self.annotations_df = self.annotations_df[self.annotations_df['label'] != -1]

    def filter_test_set(self):
        self.annotations_df = self.annotations_df[~self.annotations_df['file name'].isin(self.test_set_files)]

    def split(self, name):
        if name == 'train':
            return self.annotations_df[:int(0.8 * len(self.annotations_df))]
        elif name == 'validation':
            return self.annotations_df[int(0.8 * len(self.annotations_df)):]
        elif name == 'test':
            if self.test_split_type == TestType.ALL_TEST:
                return self.annotations_df[self.annotations_df['file name'].isin(self.test_set_files)]
            elif self.test_split_type == TestType.ONLY_PERFECT:
                return self.annotations_df[self.annotations_df['file name'].isin(self.test_set_df_only_perfect)]
            elif self.test_split_type == TestType.WITHOUT_ARTIFACTS:
                return self.annotations_df[self.annotations_df['file name'].isin(self.test_set_df_without_artifacts)]

    def add_num_samples_and_sampling_rate_per_patient(self):
        self.annotations_df['num_samples'] = self.annotations_df.apply(lambda row: parse_num_samples(row,
                                                                                                     self.dicom_dir),
                                                                       axis=1)
        self.annotations_df['sampling rate'] = self.annotations_df.apply(lambda row: parse_sampling_rate(row,
                                                                                                         self.dicom_dir)
                                                                         , axis=1)

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dicom_path = os.path.join(self.dicom_dir, f"{self.annotations_df['file name'].iloc[idx]}")
        patient_obj = patient.Patient(patient_dicom_path=dicom_path)
        if not self.threshold_35:
            echo_result = self.annotations_df['label'].iloc[idx]
        else:
            label_str = self.annotations_df['cognos_reserved_text13'].iloc[idx]
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
                  'dicom_file': self.annotations_df['file name'].iloc[idx]}
        # print(sample['ecg_signal_filtered'].shape)
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    # ds = ECGToEchoDataset(
    #     excel_path='dataset_full_details.csv', dicom_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead'
    #                                                          '_research/saar/new_data_filtered/dataset',
    #     transform=ToTensor())
    # dataloader = DataLoader(ds, batch_size=4,
    #                         shuffle=True, num_workers=1)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['ecg_signal_filterd'].size(),
    #           sample_batched['echo'].size(), sample_batched['echo'])

    # full_dataset_csv = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/dataset_full_details.csv'
    # see_below_excel = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/excel_files/echo-ecg-see-below.xlsx.xlsx'
    # fill_see_below(full_dataset_csv, see_below_excel)
    # ds = ECGToEchoDataset(
    #     excel_path=full_dataset_csv, dicom_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead'
    #                                                          '_research/saar/new_data_filtered/dataset',transform=ToTensor(), threshold_35=True)
    # for x in ds:
    #     print(x)
    #     break

    excel_file = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/full_dataset_with_see_below_2.csv'
    dicom_dir = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset'
    train_ecg_dataset = ECGToEchoDataset(excel_path=excel_file, dicom_dir=dicom_dir,
                                                             split_name='test',
                                                             transform=ToTensor(),
                                                             threshold_35=False)
    import datetime


    # i = 0
    # dates = []
    # for data in train_ecg_dataset:
    #     i +=1
    #     patient_obj = data['patient_obj']
    #     date_str = patient_obj.date
    #     date_time_obj = datetime.datetime.strptime(date_str, '%d %b %Y %H:%M')
    #     print(date_time_obj)
    #     dates.append(date_time_obj)
    #
    # print(max(dates))

    df = pd.read_csv(excel_file)
    print(df['date'].max())

    import matplotlib.pyplot as plt
    train_ecg_dataset.annotations_df['delta_days'].hist(bins=180)
    plt.show()
    print(train_ecg_dataset.annotations_df['delta_days'].value_counts())




