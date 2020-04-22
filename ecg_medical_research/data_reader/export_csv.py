"""export meta data to .csv files"""
import os
from ecg_medical_research.data_reader import patient
import pandas as pd
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Process data to csv.')
parser.add_argument('--data_dir', type=str, help='Full path to a directory '
                                               'with .dcm files.')


def export_data_to_csv(dicom_dir):
    print("Starting job..")
    dict_info = defaultdict(list)
    for f_name in os.listdir(dicom_dir):
        # print(f_name)
        if f_name.endswith('.dcm'):
            file_path = os.path.join(dicom_dir, f_name)
            print("Processing file {}...".format(f_name))
            p = patient.Patient(file_path)
            file_name = f_name
            first_name = p.first_name
            last_name = p.last_name
            patient_id = p.id
            date = " ".join(p.date.split()[:-1])
            time = p.date.split()[-1]
            print("file name: {}\nfirst name: {}\nlast name: {}\npatient id: {}\ndate: {}\ntime: {}\n".format(file_name,
                                                                                                              first_name,
                                                                                                              last_name,
                                                                                                              patient_id,
                                                                                                              date,
                                                                                                              time))
            dict_info['file name'].append(file_name)
            dict_info['first name'].append(first_name)
            dict_info['last name'].append(last_name)
            dict_info['patient id'].append(patient_id)
            dict_info['date'].append(date)
            dict_info['time'].append(time)

    df = pd.DataFrame(dict_info, columns=['file name', 'first name', 'last name', 'patient id', 'date', 'time'])
    df.to_csv('data_info.csv', index=False)
    print("Export completed...")


if __name__ == "__main__":
    args = parser.parse_args()
    dicom_dir = args.data_dir
    # dicom_dir = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data/demo_data'
    export_data_to_csv(dicom_dir)
