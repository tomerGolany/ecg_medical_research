"""Module which defines a Patient who holds 12-lead-ECG signal and additional meta data."""
from ecg_medical_research.data_reader import decode_dcom_files
from ecg_medical_research.data_reader import subsampling

class Patient(object):
    """Patient represents all the medical information within a single patient.


    Attributes:
        id: patient unique identifier.
        unfiltered_signals: 12-lead ecg signals measured from the patient.
        date: date that the ecg test was performed.
        sampling_frequency: ECG sampling frequency.
        number_of_samples: length of each ecg signal from each channel.

    """
    def __init__(self, patient_dicom_path):
        """Initialize a new Patient object, given a DCM file of a patint.

        Decodes a DCM file and stores in a patient object.

        :param patient_dicom_path: Path to a DCM path with the patient contents.
        """
        decoded_dcom = decode_dcom_files.DecodeDCOM(patient_dicom_path)
        self.filtered_signals = decoded_dcom.filtered_signals
        self.unfiltered_signals = decoded_dcom.unfiltered_signlas
        self.sampling_frequency = decoded_dcom.sampling_frequency
        self.number_of_samples = decoded_dcom.samples
        self.first_name, self.last_name = decoded_dcom.patient_name()
        self.patient_name = "{} {}".format(self.first_name, self.last_name)
        self.id = decoded_dcom.id
        self.age = decoded_dcom.age
        self.gender = decoded_dcom.gender
        self.decoded_dcom_obj = decoded_dcom
        self.birth_date = decoded_dcom.birth_date()
        self.date = decoded_dcom.date()
        # self.sub_sampled_ecg = subsampling.subsample_ecg(self.filtered_signals)
        # self.date_time

    def print_info(self):
        """Print info about the patient and about the ecg signals."""
        info = "%s\nID: %s\nGender: %s\nBirth Date: %s (Age: %s)\nAcquisition date: %s" % (
            self.patient_name,
            self.id,
            self.gender,
            self.birth_date,
            self.age,
            self.date
        )
        print(info)

    def get_signals(self, filtered=True):
        """Get 12-lead ecg signals from the patient"""
        if filtered:
            return self.filtered_signals
        else:
            return self.unfiltered_signals


