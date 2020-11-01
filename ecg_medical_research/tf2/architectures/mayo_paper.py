"""Original code which implements Mayo's paper."""
from collections import Counter
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import concatenate, Activation, Dropout, Dense, ZeroPadding2D
from keras.layers import Input, add, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM

modelName = 'EF_Model.h5'


def ReluBN(i, reluLast=True):
    # See https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
    if reluLast:
        i = BatchNormalization()(i)
        i = Activation('relu')(i)
    else:
        i = Activation('relu')(i)
        i = BatchNormalization()(i)
    return i


def ConvPoolBlockNx1(nof, width, i, reluLast=True):
    # Create a width x 1 Conv with Nof filteres, run activations and 2x1 "Max pool decimation"
    i = Conv2D(nof, (width, 1), padding='same', kernel_initializer="glorot_normal")(i)
    i = ReluBN(i, reluLast)
    i = MaxPooling2D(pool_size=(2, 1))(i)
    return i


def BuildModel(segmentLength=512, padTo=512, n_classes=2, reluLast=True):
    # Build a convolutional neural network from predefiend building box (Conv-BN-relu-pool)

    ecgInp = Input(shape=(segmentLength, 12, 1))

    if padTo > 0 and padTo > segmentLength:
        i = ZeroPadding2D(padding=((padTo - segmentLength) / 2, 0))(ecgInp)
    else:
        i = ecgInp

    inputs = ecgInp

    i = ConvPoolBlockNx1(16, 5, i, reluLast)
    i = ConvPoolBlockNx1(16, 5, i, reluLast)
    i = ConvPoolBlockNx1(32, 5, i, reluLast)
    i = MaxPooling2D(pool_size=(2, 1))(i)  # 2*2 = 4
    i = ConvPoolBlockNx1(32, 3, i, reluLast)
    i = ConvPoolBlockNx1(64, 3, i, reluLast)
    i = ConvPoolBlockNx1(64, 3, i, reluLast)
    i = MaxPooling2D(pool_size=(2, 1))(i)  # 2*2 = 4
    i = Conv2D(128, (1, 12), padding='valid', kernel_initializer="glorot_normal")(i)
    i = ReluBN(i, reluLast)

    convFeatures = Flatten()(i)
    i = Dense(64, kernel_initializer="glorot_normal")(convFeatures)
    i = ReluBN(i, reluLast)
    i = Dropout(0.5)(i)

    i = Dense(32, kernel_initializer="glorot_normal")(i)
    i = ReluBN(i, reluLast)
    i = Dropout(0.5)(i)

    i = Dense(n_classes)(i)
    out = Activation('softmax')(i)

    model = Model(inputs=inputs, outputs=[out])
    model.summary()

    opt0 = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt0, metrics=['accuracy'])

    return model


# We cannot share data but put data in here
y_train = np.load('TrainLabels.npy')[:, np.newaxis]
X_train = np.load('TrainData.npy')[:, :, :, np.newaxis]

y_val = np.load('ValidationLabels.npy')[:, np.newaxis]
X_val = np.load('ValidationData.npy')[:, :, :, np.newaxis]

N_Train = y_train.shape[0]
N_Val = y_val.shape[0]
print('Training on :' + str(N_Train) + ' and validating on :' + str(N_Val))

n_classes = 2

# Train Network
count = Counter(classes_train[:, 0])
class_weight = {}

for i in range(n_classes):
    class_weight[i] = float(N_Train) / count[i]

model = BuildModel(segmentLength=int(5000),
                   padTo=int(5120), n_classes=n_classes, reluLast=True)

earlyStopCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=9, mode='auto')
saveBestCallback = ModelCheckpoint(modelName, monitor='val_loss', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.00001)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=128, verbose=1,
                    class_weight=class_weight,
                    callbacks=[saveBestCallback, earlyStopCallback, reduceLR])
