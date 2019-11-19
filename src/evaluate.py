import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from column_names import data

testData = pd.read_csv('../datasets/emnist-balanced-test.csv', names=data)
x_test = testData.loc[:, testData.columns != 'label']
y_test = testData['label']
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

temp = np.reshape(x_test[1], (28, 28))
temp = np.uint8(temp)
print(temp.shape)
print(temp)
cv2.imshow('image', temp)
cv2.waitKey(2000)
temp = cv2.transpose(temp)
cv2.imshow('image', temp)
cv2.waitKey(2000)
temp = cv2.transpose(temp)
cv2.imshow('image', temp)
cv2.waitKey(2000)

x_test = np.reshape(x_test, (-1, 28, 28, 1))

model = tf.keras.models.load_model("logs/modelsBalancedFinal/model1.hdf5")

for i in range(len(x_test)):
    y = model.predict_classes(np.reshape(x_test[i], (-1, 28, 28, 1)))
    print("Predicted: {}  Actual: {}".format(y, y_test[i]))
    if i == 1:
        break