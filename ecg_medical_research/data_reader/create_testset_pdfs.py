"""Choose randmoly 1000 patients and copy their pdf files to a designated location."""
import pandas as pd
import argparse
import logging
import os
from shutil import copyfile

parser = argparse.ArgumentParser(description='Parse excel and find ECG and ECHO matches.', )
# parser.add_argument('--excel_path', type=str, help='Full path to excel file with ECG and ECHO.', required=True)
parser.add_argument('--dest_dir', type=str, help='Full path to destination directory.', required=True)


SICK_VALUES = ['30-35 moderate dysfunction', '35 moderate dysfunction',
               '25 severe dysfunction', '45-50 mild dysfunction',
               '30 moderate-severe dysfunction', '45  mild dysfunction',
               '20 severe dysfunction', '35-40 moderate dysfunction',
               '20-25 severe dysfunction', '40-45 mild dysfunction',
               '25-30 severe dysfunction', '40 mild-moderate dysfunction',
               '<20 severe dysfunction', 'mild dysfunction',
               'moderate dysfunction', 'severe dysfunction']

HEALTHY_VALUES = ['50-55 preserved', '60 normal', '55 preserved', '65 hyperdinamic', '50 borderline', '60-65 normal',
                  '>65 hyperdinamic', '55-60 normal']


def get_label(row):
    if row['cognos_reserved_text13'] in SICK_VALUES:
        return 0
    elif row['cognos_reserved_text13'] in HEALTHY_VALUES:
        return 1
    else:
        assert row['cognos_reserved_text13'] == 'see below'
        logging.info("Undefined label...")
        return -1


def create_testset_excel(excel_path):
    #
    # Read excel path:
    #
    df = pd.read_excel(excel_path)

    healthy_df = df[df['label'] == 1]
    healthy_df = healthy_df.reset_index()
    sick_df = df[df['label'] == 0]
    sick_df = sick_df.reset_index()
    logging.info("Number of healthy exams: %d", len(healthy_df))
    logging.info("Number of sick exams: %d", len(sick_df))

    #
    # Sample 500 from each:
    #
    sampled_healthy = healthy_df.sample(n=500)
    sampled_sick = sick_df.sample(n=500)
    concatenated_df = pd.concat([sampled_healthy, sampled_sick])
    concatenated_df = concatenated_df.sample(frac=1).reset_index(drop=True)
    concatenated_df.to_excel("test_set_v2.xlsx")
    logging.info("Completed...")


def copy_test_pdf_files(test_set_path, destination_dir):
    df = pd.read_excel(test_set_path)
    dicom_names = list(df['file name'])
    logging.info("Number of dicom file names found: %d", len(dicom_names))
    assert len(dicom_names) == len(df['file name'].unique())
    pdf_files = [f'{os.path.splitext(f)[0]}.pdf' for f in dicom_names]
    logging.info("Copying pdf files to destination %s...", destination_dir)
    num_copied = 0
    for pdf_file in pdf_files:
        if not os.path.isfile(pdf_file):
            logging.info("pdf file %s doesn't exists...", pdf_file)
            continue
        copyfile(pdf_file, os.path.join(destination_dir, f'{num_copied + 1}.pdf'))
        if os.path.isfile(os.path.join(destination_dir, f'{num_copied + 1}.pdf')):
            num_copied += 1
        else:
            logging.info("Failed to copy file %s", pdf_file)
    logging.info("Successfully copied %d files.", num_copied)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    # excel_path = args.excel_path
    dest_dir = args.dest_dir
    # if not os.path.isfile(excel_path):
    #     logging.info("%s excel file doesn't exists...exiting.", excel_path)
    #     exit(-1)
    # create_testset_excel('/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered_v2.xlsx')

    copy_test_pdf_files(test_set_path='test_set_v2.xlsx', destination_dir=dest_dir)

