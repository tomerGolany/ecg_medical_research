"""Filter ECGs according to Quality annotations."""
import enum
import pandas as pd
import logging

ECG_TEST_SET_QUALITY_FILE = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/excel_files/ecg_quality/ECG quality.xltx'
# ECG_TEST_SET_QUALITY_FILE = '../data_reader/excel_files/ecg_quality/ECG quality.xltx'


class EcgQuality(enum.Enum):
    PERFECT = "perfect"
    MINIMAL_ARTIFACTS = "minimal artifacts (1-3 leads)"
    INTERMEDIATE_ARTIFACTS = "intermediate artifacts (>3 leads)"
    SEVERE_ARTIFACTS = "severe artifacts"


# ECGS_TO_IGNORE = [545, 766, 307, 889]
ECGS_TO_IGNORE = [63, 137, 190, 270, 285, 307, 419, 545, 766, 796, 804, 833, 889, 924]

def test_set_quality_keep(ecg_type: EcgQuality):
    """Keeps only ECGs from the specified quality."""
    df = pd.read_excel(ECG_TEST_SET_QUALITY_FILE)
    assert len(df) == len(df.ecg.unique())
    logging.info("Number of ECGs (Before filtering.): %d", len(df))
    df = df[df.quality == ecg_type.value]
    logging.info("Number of ECGs %s: %d", ecg_type.value, len(df))
    return df.ecg


def test_set_quality_filter(ecg_type: EcgQuality):
    """Filter out ECGs from the specified quality."""
    df = pd.read_excel(ECG_TEST_SET_QUALITY_FILE)
    assert len(df) == len(df.ecg.unique())
    logging.info("Number of ECGs (Before filtering.): %d", len(df))
    df = df[df.quality != ecg_type.value]
    logging.info("Number of ECGs without %s: %d", ecg_type.value, len(df))
    return df.ecg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_set_quality_filter(EcgQuality.SEVERE_ARTIFACTS)