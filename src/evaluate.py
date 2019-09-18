import tensorflow as tf
import pandas as pd
import numpy as np
from column_names import data

testData = pd.read_csv('../datasets/data/emnist/emnist-balanced-test.csv', names=data)
x_test = testData.loc[:, testData.columns != 'label']
y_test = testData['label']
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

x_test = np.reshape(x_test, (y_test.shape[0], 28, 28, 1))

model = tf.keras.models.load_model("../titanic-balanced.hdf5")
model.evaluate(x_test, y_test)