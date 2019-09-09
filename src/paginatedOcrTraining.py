import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

nets = 1
model = [0] *nets
tensorboard = [0] *nets

datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.15, width_shift_range=0.1, height_shift_range=0.1)
sgd = optimizers.SGD(lr=0.001)

for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.25))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.25))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.25))
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(62, activation='softmax'))
    model[j].compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    # model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model[j].summary()
    NAME = "{}-conv-{}-nodes-{}-dense{}".format(5, 128, 3, int(time.time()))
    tensorboard[j] = TensorBoard(log_dir='logs/model1/{}'.format(NAME))

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

for trainingDataChunk in pd.read_csv('../datasets/data/trainingRecords.csv', chunksize=10000):
    for j in range(nets):
        x_train = trainingDataChunk.loc[:, trainingDataChunk.columns != 'label']
        y_train = trainingDataChunk['label']
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_train = np.reshape(x_train, (y_train.shape[0], 28, 28, 1))
        y_train = to_categorical(y_train, num_classes=62)
        X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, test_size = 0.2)
        model[j].fit(datagen.flow(X_train2, Y_train2, batch_size=64), epochs=20, validation_data=(X_val2,Y_val2), callbacks=[annealer, tensorboard[j]])

testData = pd.read_csv('../datasets/data/validationRecords.csv')
for j in range(nets):
    x_test = testData.loc[:, testData.columns != 'label']
    y_test = testData['label']
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    x_test = np.reshape(x_test, (y_test.shape[0], 28, 28, 1))
    y_test = to_categorical(y_test, num_classes=62)
    model[j].evaluate(datagen.flow(x_test, y_test))
    model[j].save("../models1/finalModel{0:d}.hdf5".format(j+1))

