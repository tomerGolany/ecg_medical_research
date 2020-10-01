import os
import logging
import pandas as pd
import argparse
import shutil


parser = argparse.ArgumentParser(description='Parse excel file and copy dicom files.', )
# parser.add_argument('--excel_path', type=str, help='Full path to excel file with dicom info.', required=True)
parser.add_argument('--source_dicom_dir', type=str, help='Full path to the source directory with the dicom files.',
                    required=False)
parser.add_argument('--dest_dicom_dir', type=str, help='Full path to the destination directory to copy the dicom files.'
                    , required=False)


def copy_dicom_files_to_destination_dir(excel_path, source_dicom_dir, destination_dir):
    """Copy all the dcm files to to the desired folder.

    :param excel_path: Path the excel path with data.
    :param destination_dir:
    :param source_dicom_dir:
    :return:
    """
    # Validate arguments:
    if not os.path.isfile(excel_path):
        logging.info("Error: Excel file %s not found... Exiting...", excel_path)
        exit(-1)
    if not os.path.isdir(source_dicom_dir):
        logging.info("Error: Source dicom dir %s doesn't exist... Exiting...", source_dicom_dir)
        exit(-1)
    if not os.path.isdir(destination_dir):
        logging.info("Destination dir doesn't exist... Exiting...", destination_dir)
        exit(-1)

    # Parse dicom file names:
    df = pd.read_excel(excel_path)
    file_names = df['file name']
    logging.info("Number of dicom files in excel: %d", len(file_names))
    file_names_unique = file_names.unique()
    logging.info("Number of unique dicom files: %d", len(file_names_unique))
    num_files_copied = 0
    for index, dicom_file_name in enumerate(file_names_unique):
        full_dicom_path = os.path.join(source_dicom_dir, dicom_file_name)
        if not os.path.isfile(full_dicom_path):
            logging.info("Dicom file %s not found", full_dicom_path)
        else:
            dest_dicom_name = dicom_file_name
            shutil.copyfile(full_dicom_path, os.path.join(dest_dicom_dir, dest_dicom_name))
            if os.path.isfile(os.path.join(dest_dicom_dir, dest_dicom_name)):
                num_files_copied += 1
        if index % 100 == 0:
            logging.info("Copied %d dicom files...", num_files_copied)
    logging.info("Successfully copied total of %d files.", num_files_copied)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    excel_path = 'echo_ecg_filtered_v2.xlsx'
    source_dicom_dir = args.source_dicom_dir
    dest_dicom_dir = args.dest_dicom_dir
    copy_dicom_files_to_destination_dir(excel_path, source_dicom_dir, dest_dicom_dir)
