import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential

trainingData = pd.read_csv('datasets/data/trainingRecords.csv')
trainingDataX = trainingData.loc[:, trainingData.columns != 'label']
trainingDataY = trainingData['label']
trainingDataX = trainingDataX.to_numpy()
trainingDataY = trainingDataY.to_numpy()
# print(trainingDataX.loc[[0]])
# print(trainingDataY[1])
print(type(trainingDataX))