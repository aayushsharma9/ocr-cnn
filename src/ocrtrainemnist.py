import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import time
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=np.float16, shape='auto'):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    if shape == 'auto':
        data = data.reshape((-1, iter_loadtxt.rowlength))
    else:
        data = data.reshape(shape)
    return data

X_train = iter_loadtxt('../datasets/xTrain.csv', shape=(-1, 28, 28, 1))
print('X_train loaded')
Y_train = iter_loadtxt('../datasets/yTrain.csv', dtype=int)
print('Y_train loaded')

X_test = iter_loadtxt('../datasets/xTest.csv', shape=(-1, 28, 28, 1))
print('X_test loaded')
Y_test = iter_loadtxt('../datasets/yTest.csv', dtype=int)
print('Y_test loaded')

print(X_test.shape, X_train.shape)
NAME = "{}-conv-{}-nodes-{}-dense{}".format(5, 128, 3, int(time.time()))
print('Name:', NAME)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.15, width_shift_range=0.1, height_shift_range=0.1)

nets = 2
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(62, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print('Model', j+1, ':')
    model[j].summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
epochs = 30
history = [0] * nets

for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))


for j in range(nets):
	    model[j].evaluate(X_test, Y_test)

for j in range(nets):
	model[j].save("../models2/finalModel{0:d}.hdf5".format(j+1))