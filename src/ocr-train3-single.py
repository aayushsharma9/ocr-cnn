import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
from column_names import data
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential

trainingData = pd.read_csv('../input/emnist/emnist-letters-train.csv', names=data)
testData = pd.read_csv('../input/emnist/emnist-letters-test.csv', names=data)
num_classes = 47
x_train = trainingData.loc[:, trainingData.columns != 'label']
y_train = trainingData['label']
x_test = testData.loc[:, testData.columns != 'label']
y_test = testData['label']
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

print(x_train.shape)
x_train = np.reshape(x_train, (y_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (y_test.shape[0], 28, 28, 1))
print(x_train.shape)

print(x_train.shape)
NAME = "{}-conv-{}-nodes-{}-dense{}".format(7, 128, 1, int(time.time()))
print('Name:', NAME)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[tensorboard])
model.evaluate(x_test, y_test)
model.save("/kaggle/working/titanic.hdf5")