import pydicom
from pydicom.data import get_testdata_files
from ecg import ECG
from docopt import docopt
from io import BytesIO
import sys
from matplotlib import pyplot as plt
from ecg_medical_research.data_reader import decode_dcom_files
from ecg_medical_research.data_reader import patient

dicom_file_path = 'ecg_medical_research/data/demo_data/ECGSample.dcm'
#
# source = BytesIO(open(dicom_file_path, mode='rb').read())
# e = ECG(source)
#
# signals = e.signals
# print(type(signals))
# print(signals.shape)
# lead_1 = signals[0]
# print(lead_1)
# print("Length of first lead: ", len(lead_1))
# plt.figure()
# plt.plot(lead_1[500:6000])
# plt.show()
#
# # print(e.channel_definitions)
#
# print(e.channels_no)
# print(e.sampling_frequency)
# print(e.samples)
# print(e.duration)
#
# interp = e.interpretation()
# print(interp)
# inf = e.print_info(interp)
# print(inf)

# decoded_dcom = decode_dcom_files.DecodeDCOM(dicom_file_path)
# unfiltered_signals = decoded_dcom.unfiltered_signlas
# filtererd_signals = decoded_dcom.filtered_signals
# plt.figure()
# plt.plot(unfiltered_signals[0][500:6000])
# plt.title("Unfiltered signals")
# plt.figure()
# plt.plot(filtererd_signals[0][500:6000])
# plt.title("Filtered signals")
# plt.show()
# print(decoded_dcom.channel_definitions)

p = patient.Patient(dicom_file_path)
p.print_info()
sigs = p.get_signals()

print("Signals shape: ", sigs.shape)
