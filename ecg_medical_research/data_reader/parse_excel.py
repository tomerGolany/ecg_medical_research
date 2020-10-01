import pandas as pd
import logging
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Parse excel file and copy dicom files.', )
parser.add_argument('--excel_path', type=str, help='Full path to excel file with dicom info.', required=True)
parser.add_argument('--source_dicom_dir', type=str, help='Full path to the source directory with the dicom files.',
                    required=True)
parser.add_argument('--dest_dicom_dir', type=str, help='Full path to the destination directory to copy the dicom files.'
                    , required=True)


def copy_dicom_files(excel_path, source_dicom_dir, dest_dicom_dir):
    """

    :param excel_path:
    :param source_dicom_dir:
    :param dest_dicom_dir:
    :return:
    """
    #
    # 1. Parse the excel file with pandas:
    #
    logging.info("Reading excel file...")
    df = pd.read_excel(excel_path)
    logging.debug("Column names: %s", ' '.join(list(df.columns)))
    logging.debug("First 5 rows in table: %s", df.head())
    files_with_echo_df = df[df['delta <180'] == 1]
    logging.info("Number of files with echo: %d", len(files_with_echo_df))
    unique_echo_patients = files_with_echo_df['patient id'].unique()
    logging.info("Number of unique patients with echo: %d", len(unique_echo_patients))
    #
    # 2. copy dicom files:
    #
    num_successful_copies = 0
    for index, row in files_with_echo_df.iterrows():
        dicom_file_name = row['file name']
        full_dicom_path = os.path.join(source_dicom_dir, dicom_file_name)
        if not os.path.isfile(full_dicom_path):
            logging.info("Dicom file %s not found", full_dicom_path)
        else:
            dest_dicom_name = 'r_{}'.format(dicom_file_name)
            shutil.copyfile(full_dicom_path, os.path.join(dest_dicom_dir, dest_dicom_name))
            if os.path.isfile(os.path.join(dest_dicom_dir, dest_dicom_name)):
                num_successful_copies += 1
        if index % 100 == 0:
            logging.info("Copied %d dicom files...", num_successful_copies)
    logging.info("Successfully copied total of %d files.", num_successful_copies)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    excel_path = args.excel_path
    source_dicom_dir = args.source_dicom_dir
    dest_dicom_dir = args.dest_dicom_dir
    if not os.path.isfile(excel_path):
        logging.info("%s excel file doesn't exists...exiting.", excel_path)
    if not os.path.isdir(source_dicom_dir):
        logging.info("Source dicom dir %s doesn't exists...exiting.", source_dicom_dir)
    else:
        logging.info("Source dicom dir: %s", source_dicom_dir)
    if not os.path.isdir(dest_dicom_dir):
        logging.info("Destination dicom dir %s doesn't exists...exiting.", dest_dicom_dir)
    copy_dicom_files(excel_path=excel_path, source_dicom_dir=source_dicom_dir, dest_dicom_dir=dest_dicom_dir)
