"""Module which decodes a DCOM file containing an ECG signal."""
import pydicom as dicom
import numpy as np
import struct
from scipy.signal import butter, lfilter
from datetime import datetime
import re


def butter_lowpass(highcut, sampfreq, order):
    """Supporting function.

    Prepare some data and actually call the scipy butter function.
    """

    nyquist_freq = .5 * sampfreq
    high = highcut / nyquist_freq
    num, denom = butter(order, high, btype='lowpass')
    return num, denom


def butter_lowpass_filter(data, highcut, sampfreq, order):
    """Apply the Butterworth lowpass filter to the DICOM waveform.

    @param data: the waveform data.
    @param highcut: the frequencies from which apply the cut.
    @param sampfreq: the sampling frequency.
    @param order: the filter order.
    """

    num, denom = butter_lowpass(highcut, sampfreq, order=order)
    return lfilter(num, denom, data)


class DecodeDCOM(object):
    def __init__(self, file_path):
        """Initialize a new DecodeDCOM object.

        :param file_path: Path to the source file.
        """
        if not isinstance(file_path, str):
            raise ValueError("file path should be a string format. instead: {}".format(file_path))
        else:
            try:
                self.dicom_obj = dicom.read_file(file_path)
                """@ivar: the dicom object."""
            except dicom.filereader.InvalidDicomError as err:
                raise Exception("Error reading dcom file: {}".format(err))

        sequence_item = self.dicom_obj.WaveformSequence[0]

        assert (sequence_item.WaveformSampleInterpretation == 'SS')
        assert (sequence_item.WaveformBitsAllocated == 16)

        self.channel_definitions = sequence_item.ChannelDefinitionSequence
        self.wavewform_data = sequence_item.WaveformData
        self.channels_no = sequence_item.NumberOfWaveformChannels
        self.samples = sequence_item.NumberOfWaveformSamples
        self.sampling_frequency = sequence_item.SamplingFrequency
        self.duration = self.samples / self.sampling_frequency
        self.filtered_signals = self._signals()

        self.age = self.dicom_obj.get('PatientAge', '').strip('Y')

        self.id = self.dicom_obj.PatientID
        self.gender = self.dicom_obj.PatientSex

    def _signals(self):
        """
        Retrieve the signals from the DICOM WaveformData object.

        sequence_item := dicom.dataset.FileDataset.WaveformData[n]

        @return: a list of signals.
        @rtype: C{list}
        """

        factor = np.zeros(self.channels_no) + 1
        baseln = np.zeros(self.channels_no)
        units = []
        for idx in range(self.channels_no):
            definition = self.channel_definitions[idx]

            assert (definition.WaveformBitsStored == 16)

            if definition.get('ChannelSensitivity'):
                factor[idx] = (
                    float(definition.ChannelSensitivity) *
                    float(definition.ChannelSensitivityCorrectionFactor)
                )

            if definition.get('ChannelBaseline'):
                baseln[idx] = float(definition.get('ChannelBaseline'))

            units.append(
                definition.ChannelSensitivityUnitsSequence[0].CodeValue
            )

        unpack_fmt = '<%dh' % (len(self.wavewform_data) / 2)
        unpacked_waveform_data = struct.unpack(unpack_fmt, self.wavewform_data)
        signals = np.asarray(
            unpacked_waveform_data,
            dtype=np.float32).reshape(
            self.samples,
            self.channels_no).transpose()

        for channel in range(self.channels_no):
            signals[channel] = (
                (signals[channel] + baseln[channel]) * factor[channel]
            )

        high = 40.0

        self.unfiltered_signlas = np.array(signals)

        # conversion factor to obtain millivolts values
        millivolts = {'uV': 1000.0, 'mV': 1.0}

        for i, signal in enumerate(signals):
            signals[i] = butter_lowpass_filter(
                np.asarray(signal),
                high,
                self.sampling_frequency,
                order=2
            ) / millivolts[units[i]]

        return signals

    def patient_name(self):
        try:
            last_name, first_name = str(self.dicom_obj.PatientName).split('^')
        except ValueError:
            last_name = str(self.dicom_obj.PatientName)
            first_name = ''
        return first_name, last_name

    def birth_date(self):
        try:
            pat_bdate = datetime.strptime(
                self.dicom_obj.PatientBirthDate, '%Y%m%d').strftime("%e %b %Y")
        except ValueError:
            pat_bdate = ""
        return pat_bdate

    def date(self):
        # Strip microseconds from acquisition date
        regexp = r"\.\d+$"
        acquisition_date_no_micro = re.sub(
            regexp, '', self.dicom_obj.AcquisitionDateTime)

        acquisition_date = datetime.strftime(
            datetime.strptime(
                acquisition_date_no_micro, '%Y%m%d%H%M%S'),
            '%d %b %Y %H:%M'
        )
        return acquisition_date
