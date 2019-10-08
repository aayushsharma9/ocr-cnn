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
from column_names import data

trainingData = pd.read_csv('../datasets/emnist-letters-train.csv', names=data)
testData = pd.read_csv('../datasets/emnist-letters-test.csv', names=data)
num_classes = 47
x_train = trainingData.loc[:, trainingData.columns != 'label']
y_train = trainingData['label']
x_test = testData.loc[:, testData.columns != 'label']
y_test = testData['label']
X_train = x_train.to_numpy()
Y_train = y_train.to_numpy()
X_test = x_test.to_numpy()
Y_test = y_test.to_numpy()

Y_train = to_categorical(Y_train, 27)
Y_test = to_categorical(Y_test, 27)

print (Y_test.shape, Y_train.shape)

X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))

datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.15, width_shift_range=0.1, height_shift_range=0.1)

nets = 2
model = [0] * nets
tensorboard = [0] * nets
csv_logger = [0] * nets

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
    model[j].add(Dense(num_classes, activation='softmax'))

    model[j].compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy', 'sparse_categorical_accuracy', 'mean_squared_error', 'mean_absolute_error', 'sparse_categorical_entropy'])

    NAME = "{}-conv-{}-nodes-{}-dense{}".format(5, 128, 3, int(time.time()))
    print('Name:', NAME)
    tensorboard[j] = TensorBoard(log_dir='logs/modelsBalanced/{}______{}'.format(NAME, j))
    csv_logger[j] = CSVLogger('./csvlogs/{}__{}.csv'.format(NAME, j), separator=',', append=False)

    print('Model', j+1, ':')
    model[j].summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
epochs = 30
history = [0] * nets

for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer, tensorboard[j]])
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))


for j in range(nets):
	    model[j].evaluate(X_test, Y_test)

for j in range(nets):
	model[j].save("logs/modelsBalanced/model{0:d}.hdf5".format(j+1))