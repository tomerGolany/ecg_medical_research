"""A script to run in the Hospital Computers, to find new dicom files.

It is assumed that the script is ran inside a directory with DICOM files.

"""
import pandas as pd
import os
import logging
from ecg_medical_research.data_reader import patient
import datetime


def export_excel_with_new_dicoms(minimum_date, dicom_dir='.'):
    """Export an excel with lists of all dicom files.

    The excel holds names of dicom files with thier ID numbers and date.

    :param minimum_date:
    :param dicom_dir:
    :return:
    """

    dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    logging.info("Found %d dcm files...", len(dcm_files))
    dicom_paths = []
    ecg_dates = []
    patient_ids = []
    for f in dcm_files:
        dicom_path = os.path.join(dicom_dir, f)
        patient_obj = patient.Patient(patient_dicom_path=dicom_path)
        ecg_date = patient_obj.date
        date_time_obj = datetime.datetime.strptime(ecg_date, '%d %b %Y %H:%M')
        if date_time_obj > minimum_date:
            logging.info("Found new ECG. DICOM: %s. ID: %s. Date: %s", f, patient_obj.id, date_time_obj)
            dicom_paths.append(dicom_path)
            ecg_dates.append(date_time_obj)
            patient_ids.append(patient_obj.id)

    df = pd.DataFrame(list(zip(dicom_paths, ecg_dates, patient_ids)),
                      columns=['DICOM_PATH', 'ECG_DATE', 'PATIENT_ID'])
    df.to_csv("new_ecgs.csv")


if __name__ == "__main__":
    minimum_date = '28 APR 2020 00:00'
    # minimum_date = '2020-04-28 00:00:00'
    minimum_date = datetime.datetime.strptime(minimum_date, '%d %b %Y %H:%M')
    print(minimum_date)
    export_excel_with_new_dicoms(minimum_date, dicom_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset')