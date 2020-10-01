import os
import logging
from pydicom.filereader import dcmread
import argparse


parser = argparse.ArgumentParser(description='annonimyze dicom files.', )
parser.add_argument('--dicom_dir', type=str, help='Full path to the source directory with the dicom files.',
                    required=True)


def annonimyze(dicom_dir):
    if not os.path.isdir(dicom_dir):
        logging.info("Dicom directory %s doesn't exists...exiting.", dicom_dir)

    dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    logging.info("Found %d dcm files...", len(dcm_files))
    for f in dcm_files:
        dataset = dcmread(os.path.join(dicom_dir, f), force=True)
        dataset.PatientName = 'ann'
        dataset.PatientID = f
        dataset.save_as(os.path.join(dicom_dir, f))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    dicom_dir = args.dicom_dir
    annonimyze(dicom_dir)

