import sys
import tensorflow as tf
from ecg_medical_research.tf2.architectures import antonio_paper
from ecg_medical_research.data_reader.dataset import ecg_to_echo_dataset
import argparse
import pandas as pd
import h5py
# import tensorflow_io as tfio
import os
from ecg_medical_research.data_reader import patient
import glob
from ecg_medical_research.tf2.datasets import dataset


def parse_dicom(dicom_file, dicom_dir):
    dicom_path = os.path.join(dicom_dir, f"{dicom_file}")
    patient_obj = patient.Patient(patient_dicom_path=dicom_path)
    return patient_obj.filtered_signals


def create_tf_dataset(excel_path, dicom_dir, split_name):
    def _parse_dicom(dicom_file, label):
        filtered_signals = tf.numpy_function(parse_dicom, inp=[dicom_file, dicom_dir], Tout=tf.float32)
        # dicom_path = os.path.join(dicom_dir, f"{dicom_file}")
        # patient_obj = patient.Patient(patient_dicom_path=dicom_path)
        return filtered_signals, label

    ds_pytorch = ecg_to_echo_dataset.ECGToEchoDataset(excel_path, dicom_dir, split_name, None)
    dicom_files = ds_pytorch.annotations_df['file name']
    labels = ds_pytorch.annotations_df['label']
    signals = []
    for dicom_file in dicom_files:
        dicom_path = os.path.join(dicom_dir, f"{dicom_file}")
        patient_obj = patient.Patient(patient_dicom_path=dicom_path)
        signals.append(patient_obj.filtered_signals)
    tf_ds = tf.data.Dataset.from_tensor_slices((signals, labels))
    return tf_ds


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    # parser.add_argument('path_to_hdf5', type=str,
    #                     help='path to hdf5 file containing tracings')
    # parser.add_argument('path_to_csv', type=str,
    #                     help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()

    args.path_to_csv = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/antonio_paper/sample_data/data/annotations/gold_standard.csv'
    args.path_to_hdf5 = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/antonio_paper/sample_data/data/ecg_tracings.hdf5'

    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 2
    opt = tf.keras.optimizers.Adam(lr)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=7,
                                                      min_lr=lr / 100),
                 tf.keras.callbacks.EarlyStopping(patience=9,
                                                  # Patience should be larger than the one in ReduceLROnPlateau
                                                  min_delta=0.00001)]
    # Set session and compile model
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))
    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = antonio_paper.build_model()
    model.compile(loss=loss, optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy])
    # Get annotations
    # y = pd.read_csv(args.path_to_csv).values
    # Get tracings
    # x = tfio.IODataset.from_hdf5(args.path_to_hdf5, args.dataset_name)
    # f = h5py.File(args.path_to_hdf5, "r")
    # x = f[args.dataset_name]

    # excel_path = '/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/dataset_full_details.csv'
    # dicom_dir = '/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/saar/new_data_filtered/dataset'
    # split_name = None
    # ds = create_tf_dataset(excel_path, dicom_dir, split_name)
    model_dir = 'model_outputs'
    files = glob.glob("/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/tf2/datasets/*.record")
    # print(files)
    # print(files)
    ds = dataset.create_tf_dataset(filenames=files)
    ds = ds.shuffle(buffer_size=1)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    # for raw_record in ds.take(1):
    #     print(repr(raw_record['ecg_signal']))
    # Create log
    # callbacks += [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logiss'), batch_size=batch_size, write_graph=False),
    #               tf.keras.callbacks.CSVLogger('training.log',
    #                                            append=False)]  # Change append to true if continuing training
    # # Save the BEST and LAST model
    # # callbacks += [tf.keras.callbacks.ModelCheckpoint('./backup_model_last.hdf5'),
    # #               tf.keras.callbacks.ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
    # # Train neural network
    history = model.fit(ds,
                        steps_per_epoch=10,
                        epochs=1,
                        # initial_epoch=0,  # If you are continuing a interrupted section change here
                        # validation_split=args.val_split,
                        # shuffle='batch',  # Because our dataset is an HDF5 file
                        # callbacks=callbacks,
                        # verbose=1)
                        )
    # Save final result
    # model.save("./final_model.hdf5")
    # f.close()



