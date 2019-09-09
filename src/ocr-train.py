import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
# mnist = tf.keras.datasets.mnist
import tensorflow_federated as tff
mnist = tff.simulation.datasets.emnist


(train, test) = mnist.load_data()

training = train.create_tf_dataset_from_all_clients()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = np.reshape(x_train, (y_train.shape[0], 28, 28, 1))
# x_test = np.reshape(x_test, (y_test.shape[0], 28, 28, 1))

print(training)
print(test)

# x_train, x_test = x_train / 255.0, x_test / 255.0

# # cv2.imshow("image", x_train[0])
# # cv2.waitKey(2000)

# print(x_train.shape)
# NAME = "{}-conv-{}-nodes-{}-dense{}".format(3, 128, 3, int(time.time()))
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[tensorboard])
# model.evaluate(x_test, y_test)
# model.save("../model5epochs.hdf5")
