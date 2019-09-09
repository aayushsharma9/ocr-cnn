import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

trainingData = pd.read_csv('../datasets/data/trainingRecords.csv')
testData = pd.read_csv('../datasets/data/validationRecords.csv')
x_train = trainingData.loc[:, trainingData.columns != 'label']
y_train = trainingData['label']
x_test = testData.loc[:, testData.columns != 'label']
y_test = testData['label']
# print(x_train.loc[[0]])
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

print(x_train.shape)
x_train = np.reshape(x_train, (y_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (y_test.shape[0], 28, 28, 1))
print(x_train.shape)

# x_train, x_test = x_train / 255.0, x_test / 255.0

# cv2.imshow("image", x_train[0])
# cv2.waitKey(2000)

# y_binary = to_categorical(y_train)

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# nets = 15
# model = [0] *nets
# for j in range(nets):
#     model[j] = Sequential()

#     model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
#     model[j].add(BatchNormalization())
#     model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
#     model[j].add(BatchNormalization())
#     model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
#     model[j].add(BatchNormalization())
#     model[j].add(Dropout(0.4))

#     model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
#     model[j].add(BatchNormalization())
#     model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
#     model[j].add(BatchNormalization())
#     model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
#     model[j].add(BatchNormalization())
#     model[j].add(Dropout(0.4))

#     model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
#     model[j].add(BatchNormalization())
#     model[j].add(Flatten())
#     model[j].add(Dropout(0.4))
#     model[j].add(Dense(10, activation='softmax'))
#     model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     model[j].summary()

    # NAME = "{}-conv-{}-nodes-{}-dense{}".format(5, 128, 3, int(time.time()))
    # tensorboard = TensorBoard(log_dir='logs/model1/{}'.format(NAME))
#     model[j].fit(x_train, y_binary, epochs=45, batch_size=64, validation_split=0.2, callbacks=[tensorboard, annealer])
    
#     model[j].evaluate(x_test, y_test)
#     model[j].save("../models1/finalModel{0:d}.hdf5".format(j+1))


























model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=5, activation='relu', padding='same', strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same', strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(filters=128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(62, activation='softmax'))

















# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(384, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(192, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(62, activation='softmax'))
NAME = "{}-conv-{}-nodes-{}-dense{}".format(5, 128, 3, int(time.time()))
tensorboard = TensorBoard(log_dir='logs/model1/{}'.format(NAME))


from tensorflow.keras import optimizers
sgd = optimizers.SGD(lr=0.1)

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=62)
y_test = to_categorical(y_test, num_classes=62)
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[tensorboard])
model.evaluate(x_test, y_test)
model.save("../finalModel10.hdf5")
