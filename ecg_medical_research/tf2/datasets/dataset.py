"""Create tf.dataset to iterate on 12-lead ecg and echo outcome."""
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
import numpy as np
from ecg_medical_research.data_reader.dataset import ecg_to_echo_dataset
from ecg_medical_research.data_reader import patient
import os
import tqdm
import glob


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(ecg_signal: np.ndarray, echo_label: int):
    """Creates a tf.Example message ready to be written to a file."""
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
      'ecg_signal': _bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(ecg_signal, dtype=tf.float32))),
      'label': _int64_feature(echo_label),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def decode_example(serialized_example):
    return tf.train.Example.FromString(serialized_example)


def create_tf_records(filename, excel_path, dicom_dir, split_name):
    with tf.io.TFRecordWriter(filename) as writer:
        ds_pytorch = ecg_to_echo_dataset.ECGToEchoDataset(excel_path, dicom_dir, split_name, None)
        dicom_files = ds_pytorch.annotations_df['file name']
        labels = ds_pytorch.annotations_df['label']

        for dicom_file, label in zip(dicom_files, labels):
            print(dicom_file, label)
            dicom_path = os.path.join(dicom_dir, f"{dicom_file}")
            patient_obj = patient.Patient(patient_dicom_path=dicom_path)
            example = serialize_example(patient_obj.filtered_signals, label)
            writer.write(example)


def write_shard_records(filename, excel_path, dicom_dir, split_name):
    ds_pytorch = ecg_to_echo_dataset.ECGToEchoDataset(excel_path, dicom_dir, split_name, None)
    dicom_files = ds_pytorch.annotations_df['file name']
    labels = ds_pytorch.annotations_df['label']
    index = 0
    n_ecg_shard = 300
    n_shards = len(dicom_files) // 300

    # tqdm is an amazing package that if you don't know yet you must check it
    for shard in tqdm.tqdm(range(n_shards)):
        # The original tfrecords_path is "{}_{}_{}.records" so the first parameter is the name of the dataset,
        # the second is "train" or "val" or "test" and the last one the pattern.
        tfrecords_shard_path = "{}_{}_{}.record".format(filename, "test", '%.5d-of-%.5d' % (shard, n_shards - 1))
        end = index + n_ecg_shard if len(dicom_files) > (index + n_ecg_shard) else -1
        ecg_shard_list = dicom_files[index:end]
        labeld_shard_list = labels[index:end]
        print(index, end)
        with tf.io.TFRecordWriter(os.path.join('data', tfrecords_shard_path)) as writer:
            for dicom_file, label in zip(ecg_shard_list, labeld_shard_list):
                print(dicom_file, label)
                dicom_path = os.path.join(dicom_dir, f"{dicom_file}")
                patient_obj = patient.Patient(patient_dicom_path=dicom_path)
                example = serialize_example(patient_obj.filtered_signals, label)
                writer.write(example)
        index = end


# sig = np.zeros([5499, 12])
# ex = serialize_example(sig, 0)
# ex_d = decode_example(ex)
# # print(tf.io.parse_tensor(ex_d.features.feature['ecg_signal']))
# print(ex_d.features.feature['label'].int64_list.value)
# dsig = ex_d.features.feature['ecg_signal'].bytes_list.value
# print(tf.io.parse_tensor(dsig[0], out_type=tf.float32))
# # print(ex_d.feature['label'])
#
# excel_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/full_dataset_with_see_below.csv'
# dicom_dir = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset'
# filename = 'train.tfrecord'
# # create_tf_records(filename, excel_path, dicom_dir, split_name=None)
# write_shard_records(filename, excel_path, dicom_dir, split_name=None)

feature_description = {
    'ecg_signal': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def create_tf_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)

    def _parse_example(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        parsed['ecg_signal'] = tf.io.parse_tensor(parsed['ecg_signal'], out_type=tf.float32)
        ecg = tf.transpose(parsed['ecg_signal'], perm=[1, 0])
        one_hot_label = tf.one_hot(tf.cast(parsed['label'], dtype=tf.int64), depth=2)
        return ecg, one_hot_label

    parsed_dataset = raw_dataset.map(_parse_example)
    return parsed_dataset


# files = glob.glob("*.record")
# # print(files)
# ds = create_tf_dataset(filenames=files)
# for ecg, one_hot_label in ds.take(10):
#     print(repr(one_hot_label))

# if __name__ == "__main__":
#     excel_path = '../../data_reader/dataset/full_dataset_with_see_below.csv'
#     dicom_dir = '/home/tomer.golany/dataset'
#     filename = 'test.tfrecord'
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     with tf.device('/device:GPU:3'):
#         write_shard_records(filename, excel_path, dicom_dir, split_name='test')