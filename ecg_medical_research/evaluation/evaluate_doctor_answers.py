"""Parse and evaluate doctors answers."""
import pandas as pd
import numpy as np
import logging

from ecg_medical_research.evaluation import quality

DOCTOR_ANSWERS_COMPLETENCE = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/doctors_answers/Doctor_Answers_complitions.xlsx'
DOCTOR_ANSWERS = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/doctors_answers/ECG_ECHO_FORM (Responses) (1).xlsx'


def eval_natalia(ecg_quality_keep=None, ecg_quality_filter=None):
    #
    # Read Test-Set:
    #
    test_set_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/test_set_v2.xlsx'
    test_df = pd.read_excel(test_set_path)
    ecg_ids = set(np.arange(1, 1001, 1))
    test_df['ECG number'] = ecg_ids
    test_df = test_df[['ECG number', 'label']]

    execel_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/doctors_answers/ECG_echo_table Natalia.xlsx'
    df = pd.read_excel(execel_path)
    df = df[df['EF>=50% (mark 1 if yes!)'] != '?']
    df['ECG number'] = df['no']
    logging.info("Total number of answers to evaluate: %d", len(df))
    merged_df = pd.merge(test_df, df, on=['ECG number'])
    #
    # Keep only ECGs at a quality:
    #
    if ecg_quality_keep is not None:
        quality_ecg_numbers = quality.test_set_quality_keep(ecg_quality_keep)
        merged_df = merged_df[merged_df['ECG number'].isin(quality_ecg_numbers)]
        logging.info("Number of ECGs to evaluate after keeping on quality %s: %d", ecg_quality_keep.value,
                     len(merged_df))
    if ecg_quality_filter is not None:
        quality_ecg_numbers = quality.test_set_quality_filter(ecg_quality_filter)
        merged_df = merged_df[merged_df['ECG number'].isin(quality_ecg_numbers)]
        logging.info("Number of ECGs to evaluate after keeping on quality %s: %d", ecg_quality_filter.value,
                     len(merged_df))

    predictions = list(merged_df['EF>=50% (mark 1 if yes!)'])
    ground_truths = list(merged_df['label'])
    tp, fp, tn, fn = perf_measure(ground_truths, predictions)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    logging.info(f"Doctor Name: Natalia\nScores: tpr: {tpr}. fpr: {fpr}\n\n")
    return 'Natalia', tpr, fpr, tp, fp, tn, fn, f1_score, len(ground_truths)


def eval(output_csv_name, ecg_quality_keep=None, ecg_quality_filter=None):
    exported_csv = []
    df = pd.read_excel(DOCTOR_ANSWERS)
    df_missing_answers = pd.read_excel(DOCTOR_ANSWERS_COMPLETENCE)
    df['Personal ID'] = df['Personal ID'].apply(lambda x: x.lower().replace(' ', ''))
    df_missing_answers['Personal ID'] = df_missing_answers['Personal ID'].apply(lambda x: x.lower().replace(' ', ''))
    doctor_ids = df['Personal ID'].unique()
    logging.info("#%d doctors:\n %s", len(doctor_ids), "\n".join(doctor_ids))


    #
    # Read Test-Set:
    #
    test_set_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/test_set_v2.xlsx'
    test_df = pd.read_excel(test_set_path)
    ecg_ids = set(np.arange(1, 1001, 1))
    test_df['ECG number'] = ecg_ids
    test_df = test_df[['ECG number', 'label']]

    grouped = df.groupby(['Personal ID'])
    for doctor_name, groupd_df in grouped:
        if doctor_name == 'nako':
            continue
        logging.info("Evaluating answers from Doctor %s", doctor_name)
        ecg_numbers = groupd_df['ECG number']

        #
        # Filter ECG numbers not between 1-1000:
        #
        logging.info("Filtering ECGs that are not between 1 - 1000...")
        undefined_ecg_numbers = groupd_df[~groupd_df['ECG number'].isin(ecg_ids)]
        logging.info("Undefined ECGs: %s", list(undefined_ecg_numbers['ECG number']))
        groupd_df = groupd_df[groupd_df['ECG number'].isin(ecg_ids)]

        #
        # Find ids not annotated:
        #
        missing_ids = ecg_ids - set(ecg_numbers.unique())
        logging.info("Number of unique ECG answers: %d. Missing ECGs: %s", len(ecg_numbers.unique()), missing_ids)

        #
        # Find duplicate annotations:
        #
        duplicate_ecg_answers = pd.concat(g for _, g in groupd_df.groupby("ECG number") if len(g) > 1)
        logging.info("Number of duplicate ECGs: %d. Duplicates ECGs: %s", len(duplicate_ecg_answers['ECG number'].unique()),
                     duplicate_ecg_answers['ECG number'].unique())

        groupd_df = groupd_df.drop_duplicates(subset=['ECG number'], keep=False)
        logging.info("Total number of answers to evaluate: %d", len(groupd_df))

        #
        # Search for the missing answers in the new excel file:
        #
        logging.info("Searching for the missing answers in the new excel file...")
        if doctor_name == 'reyu':
            df_more_answers = df_missing_answers[df_missing_answers['Personal ID'] == 'revi']
        else:
            df_more_answers = df_missing_answers[df_missing_answers['Personal ID'] == doctor_name]
        logging.info("Found extra answers: %s", df_more_answers['ECG number'].unique())
        groupd_df = pd.concat([groupd_df, df_more_answers])
        missing_ids = ecg_ids - set(groupd_df['ECG number'].unique()) - set(quality.ECGS_TO_IGNORE)
        logging.info("Number of unique ECG answers: %d. Missing ECGs: %s", len(groupd_df['ECG number'].unique()),
                     missing_ids)
        assert len(groupd_df['ECG number'].unique()) == len(groupd_df['ECG number'])

        #
        # Remove bad ECGs
        #
        groupd_df = groupd_df[~groupd_df['ECG number'].isin(quality.ECGS_TO_IGNORE)]
        logging.info("After removing bad ECGs: %d", len(groupd_df['ECG number'].unique()))

        answers_df = groupd_df[['ECG number', 'Is EF  equal or more than 50%']]
        answers_df = answers_df.sort_values(by=['ECG number'])
        assert answers_df['Is EF  equal or more than 50%'].unique().sort() == ['YES', 'NO'].sort()
        answers_df['Is EF  equal or more than 50%'] = answers_df['Is EF  equal or more than 50%'].apply(
            lambda x: 1 if x == 'YES' else 0)
        merged_df = pd.merge(test_df, answers_df, on=['ECG number'])
        logging.info(type(merged_df))
        #
        # Keep only ECGs at a quality:
        #
        if ecg_quality_keep is not None:
            quality_ecg_numbers = quality.test_set_quality_keep(ecg_quality_keep)
            merged_df = merged_df[merged_df['ECG number'].isin(quality_ecg_numbers)]
            logging.info(type(merged_df))
            logging.info("Number of ECGs to evaluate after keeping on quality %s: %d", ecg_quality_keep.value, len(merged_df))
        if ecg_quality_filter is not None:
            quality_ecg_numbers = quality.test_set_quality_filter(ecg_quality_filter)
            merged_df = merged_df[merged_df['ECG number'].isin(quality_ecg_numbers)]
            logging.info(type(merged_df))
            logging.info("Number of ECGs to evaluate after filtering quality %s: %d", ecg_quality_filter.value,
                         len(merged_df))

        predictions = list(merged_df['Is EF  equal or more than 50%'])
        ground_truths = list(merged_df['label'])
        tp, fp, tn, fn = perf_measure(ground_truths, predictions)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        f1_score = tp / (tp + 0.5 * (fp + fn))
        logging.info(f"Doctor Name: {doctor_name}\nScores: tpr: {tpr}. fpr: {fpr}\n\n")
        exported_csv.append((doctor_name, tpr, fpr, tp, fp, tn, fn, f1_score, len(ground_truths)))

    exported_csv.append(eval_natalia(ecg_quality_keep, ecg_quality_filter))

    exported_csv_df = pd.DataFrame(exported_csv, columns=['Name', 'True Positive Rate', 'False Positive Rate',
                                                           '#TP', '#FP', '#TN', '#FN', 'F1 Score', 'Number of ECGs'])
    print(exported_csv_df)
    exported_csv_df.to_csv(output_csv_name, index=False)


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    eval(output_csv_name='evaluations_results_all_test_set.csv')

    eval(ecg_quality_keep=quality.EcgQuality.PERFECT,
         output_csv_name='evaluations_results_only_perfect_ecgs.csv')

    eval(ecg_quality_filter=quality.EcgQuality.SEVERE_ARTIFACTS,
         output_csv_name='evaluations_results_without_severe_ecgs.csv')