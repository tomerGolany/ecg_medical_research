import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from ecg_medical_research.tf2.architectures import resnet50v2
from ecg_medical_research.tf2.architectures import antonio_paper
import glob
from ecg_medical_research.tf2.datasets import dataset
import os
import numpy as np


def train():
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lr = 0.0001
    batch_size = 64
    opt = tf.keras.optimizers.Adam(lr)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=7,
                                                      min_lr=lr / 100),
                 tf.keras.callbacks.EarlyStopping(patience=9,
                                                  # Patience should be larger than the one in ReduceLROnPlateau
                                                  min_delta=0.00001)]
    model_dir = 'model_outputs'
    callbacks += [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs_resnet50_3'), write_graph=False),
                  tf.keras.callbacks.CSVLogger('training.log', append=False)]


    files = glob.glob(
        "datasets/data/train*.record")
    print(files)
    ds = dataset.create_tf_dataset(filenames=files)
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    # ds = ds.repeat()
    test_files = glob.glob("datasets/data/test*.record")
    print(test_files)
    test_ds = dataset.create_tf_dataset(filenames=test_files)
    test_ds = test_ds.batch(batch_size)

    # model = antonio_paper.build_model()
    model = resnet50v2.ResNet50V2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(5499, 12),
    pooling=None,
    classes=2,
    classifier_activation=None)
    model.compile(loss=loss, optimizer=opt, metrics=[tf.keras.metrics.CategoricalAccuracy()])  # metrics=[tf.keras.metrics.BinaryAccuracy]

    model.fit(ds, epochs=10,  callbacks=callbacks, validation_data=test_ds)


if __name__ == "__main__":
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    # with tf.device('/device:GPU:3'):
    train()