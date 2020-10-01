"""Parse ECG and ECHO excel file to find patients who had both."""
import pandas as pd
import argparse
import logging
import os

parser = argparse.ArgumentParser(description='Parse excel and find ECG and ECHO matches.', )
parser.add_argument('--excel_path', type=str, help='Full path to excel file with ECG and ECHO.', required=True)

SICK_VALUES = ['30-35 moderate dysfunction', '35 moderate dysfunction',
               '25 severe dysfunction', '45-50 mild dysfunction',
               '30 moderate-severe dysfunction', '45  mild dysfunction',
               '20 severe dysfunction', '35-40 moderate dysfunction',
               '20-25 severe dysfunction', '40-45 mild dysfunction',
               '25-30 severe dysfunction', '40 mild-moderate dysfunction',
               '<20 severe dysfunction', 'mild dysfunction',
               'moderate dysfunction', 'severe dysfunction', '40% mild-moderate LV dysfunction',
               '35% moderate dysfunction', '35-40% moderate dysfunction', '45-50% mild LV dysfunction',
               '30% moderate-severe', '40-45% mild LV dysfunction', '25% severe LV dysfunction',
               '45% mild LV dysfunction']

HEALTHY_VALUES = ['50-55 preserved', '55% preserved', '60 normal', '60% normal',
                  '55 preserved', '65 hyperdinamic', '50 borderline', '60-65 normal',
                  '>65 hyperdinamic', '55-60 normal', '50% borderline', '50-55% preserved']

########### Threshold 35 ###############
SICK_VALUES_35 = [
            '30-35 moderate dysfunction', '35 moderate dysfunction',
               '25 severe dysfunction',
               '30 moderate-severe dysfunction',
               '20 severe dysfunction',
               '20-25 severe dysfunction',
               '25-30 severe dysfunction',
               '<20 severe dysfunction',

               '35% moderate dysfunction',
               '30% moderate-severe',  '25% severe LV dysfunction',
               ]

HEALTHY_VALUES_35 = ['40-45% mild LV dysfunction', '35-40% moderate dysfunction', '45-50% mild LV dysfunction',
    '45% mild LV dysfunction', '50-55 preserved', '55% preserved', '60 normal', '60% normal',
                  '55 preserved', '65 hyperdinamic', '50 borderline', '60-65 normal',
                  '>65 hyperdinamic', '55-60 normal', '50% borderline', '50-55% preserved']
################################################


def get_label(row):
    if row['cognos_reserved_text13'] in SICK_VALUES:
        return 0
    elif row['cognos_reserved_text13'] in HEALTHY_VALUES:
        return 1
    else:
        print("Label: ", row['cognos_reserved_text13'])
        assert row['cognos_reserved_text13'] == 'see below'
        logging.info("Undefined label...")
        return -1


def filter_by_days_fn(row):
    """Leave only rows where the distance between the ECG and ECHO is less then 180 days."""
    delta_days = abs((row['Procedure date'] - row['date']).days)
    return delta_days <= 180


def find_ecg_echo_tuples(excel_path):
    """Find tuples of ECG and ECHO exams which belong to the same patient.

    :param excel_path: Path to the excel file.
    :return: Pandas data-frame with the couples.
    """
    logging.info("Reading excel file...")
    echo_df = pd.read_excel(excel_path, sheet_name='Echo with classification')
    ecg_df = pd.read_excel(excel_path, sheet_name='ECG data_info')
    merged_df = pd.merge(left=ecg_df, right=echo_df, left_on=['patient id'], right_on=['Old Patient ID'], how='inner')
    #
    # Filter out exams which have distance > 180 days:
    #
    matches = merged_df.apply(filter_by_days_fn, axis=1)
    ecg_echo_valid_couples_df = merged_df[matches]
    logging.info("Total number of matches (Include duplications): %d", len(ecg_echo_valid_couples_df))
    logging.info("Unique patient IDs: %d", len(ecg_echo_valid_couples_df['patient id'].unique()))
    logging.info("Unique Echo procedures: %s", len(ecg_echo_valid_couples_df['Procedure ID'].unique()))
    #
    # Group by Echo and take only the ecg most close:
    #
    ecg_echo_valid_couples_df['delta_days'] = ecg_echo_valid_couples_df.apply(lambda row: abs((row['Procedure date'] -
                                                                                               row['date']).days),
                                                                              axis=1)
    grouped_by_echo = ecg_echo_valid_couples_df.groupby('Procedure ID')
    result_list = []
    for echo_id, df in grouped_by_echo:
        df = df.reset_index()
        echo_ecg_row = df.iloc[df['delta_days'].idxmin()]
        result_list.append(echo_ecg_row)
    result_df = pd.concat(result_list, axis=1).T
    #
    # Add columns with binary score 1 - Healthy. 0 - Sick.
    #
    result_df['label'] = result_df.apply(get_label, axis=1)

    result_df = filter_same_ecg_with_multiple_echos(result_df)

    num_sick = len(result_df[result_df['label'] == 0])
    num_health = len(result_df[result_df['label'] == 1])
    logging.info("Number of sick: %d, Number of healthy: %d", num_sick, num_health)

    result_df.to_excel("echo_ecg_filtered_v2.xlsx")

    logging.info("Completed...")
    return ecg_echo_valid_couples_df


def filter_same_ecg_with_multiple_echos(echo_ecg_filtered_df):
    """Filter out multiple rows which contains the same ECG exam with different echo exam.
        In Case of conflicts removes all rows.
    :param echo_ecg_filtered_df: dataframe with ecg-echo matches.
    :return: filtered dataframe.
    """
    number_of_ecg_with_multiple_echos = 0
    total_number_of_echos_with_same_ecg = 0
    num_conflicts = 0
    conflict_dfs = []
    conflict_files = []
    for file_name, duplicates_df in echo_ecg_filtered_df.groupby('file name'):
        if len(duplicates_df) > 1:
            number_of_ecg_with_multiple_echos += 1
            #
            # Check Conflicts in echo result:
            #
            echo_results = duplicates_df['label'].unique()
            if len(echo_results) != 1:
                num_conflicts += 1
                logging.info("Found conflict in echo results:\n %s", duplicates_df)
                conflict_dfs.append(duplicates_df)
                conflict_files.append(file_name)
            total_number_of_echos_with_same_ecg += len(duplicates_df)
    logging.info("Total number of ECGs with multiple echos: %d.", number_of_ecg_with_multiple_echos)
    logging.info("Total number of Echos with same ECGs: %d.", total_number_of_echos_with_same_ecg)
    logging.info("Total number of conflicts: %d", num_conflicts)
    conflicts_df = pd.concat(conflict_dfs)
    # conflicts_df.to_excel("conflicts.xlsx")
    #
    # Remove conflicts from dataframe:
    #
    logging.info("Number of files before removing conflicts: %d", len(echo_ecg_filtered_df))
    echo_ecg_filtered_df = echo_ecg_filtered_df[~echo_ecg_filtered_df['file name'].isin(conflict_files)]
    logging.info("Number of files after removing conflicts: %d", len(echo_ecg_filtered_df))
    # Verify zero conflicts:
    for file_name, duplicates_df in echo_ecg_filtered_df.groupby('file name'):
        if len(duplicates_df) > 1:
            echo_results = duplicates_df['label'].unique()
            if len(echo_results) != 1:
                logging.info("Found conflict in echo results:\n %s", duplicates_df)
    # Remove other dups:
    echo_ecg_filtered_df = echo_ecg_filtered_df.drop_duplicates(subset='file name', keep='first')
    # Verify zero duplicates
    assert len(echo_ecg_filtered_df) == len(echo_ecg_filtered_df['file name'].unique())
    logging.info("Number of files after removing duplications: %d", len(echo_ecg_filtered_df))
    return echo_ecg_filtered_df


def echo_values():
    """Print echo values from the filtered excel."""
    filtered_excel_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/echo_ecg_filtered.xlsx'
    echo_df = pd.read_excel(filtered_excel_path)
    echo_results = echo_df['cognos_reserved_text13'].unique()
    logging.info("Echo unique values: %s", echo_results)
    # echo_with_see_below = echo_df[echo_df['cognos_reserved_text13'] == 'see below']
    # print(len(echo_with_see_below))
    # echo_with_see_below.to_excel("echo_ecg_see_below.xlsx")

    # Find multiple echos belonging to the same ECG (same dicom file):
    # file_names_df = echo_df['file name']
    # duplicated_file_names = file_names_df[file_names_df.duplicated()]
    # logging.info("Number of duplicated file names: %d", len(duplicated_file_names))
    # df_with_duplicated_file_names = echo_df[echo_df['file name'].isin(duplicated_file_names)].sort_values('file name')
    # logging.info("Length of duplicated files df: %s", len(df_with_duplicated_file_names))
    number_of_ecg_with_multiple_echos = 0
    total_number_of_echos_with_same_ecg = 0
    conflict_file_names = []
    for file_name, duplicates_df in echo_df.groupby('file name'):
        if len(duplicates_df) > 1:
            number_of_ecg_with_multiple_echos += 1
            logging.info("Found ECG with multiple Echos: %s.", file_name)
            logging.info("Number of echos: %d.", len(duplicates_df))
            #
            # Check Conflicts in echo result:
            #
            echo_results = duplicates_df['label'].unique()
            if len(echo_results) != 1:
                logging.info("Found conflict in echo results:\n %s", duplicates_df)
                conflict_file_names.append(file_name)
            total_number_of_echos_with_same_ecg += len(duplicates_df)
    logging.info("Total number of ECGs with multiple echos: %d.", number_of_ecg_with_multiple_echos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    excel_path = args.excel_path
    if not os.path.isfile(excel_path):
        logging.info("%s excel file doesn't exists...exiting.", excel_path)
        exit(-1)
    find_ecg_echo_tuples(excel_path)
    # echo_values()
