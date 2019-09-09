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

trainingData = pd.read_csv('../datasets/train.csv')
testData = pd.read_csv('../datasets/test.csv')
X_train = trainingData.loc[:, trainingData.columns != 'label']
Y_train = trainingData['label']
X_test = testData.loc[:, testData.columns != 'label']
Y_test = testData['label']
trainingData = []
testData =  []
X_train = X_train.to_numpy()
# X_train /= 255
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
# X_test /= 255
Y_test = Y_test.to_numpy()

print(X_test.shape, X_train.shape)

# X_train = np.reshape(X_train, (-1, 28, 28, 1))
# X_test = np.reshape(X_test, (-1, 28, 28, 1))

# Y_train = to_categorical(Y_train, num_classes=62)

# # print(Y_train.shape)
# # print(Y_train[0])
# # print(Y_train[1])
# # print(Y_train)

# NAME = "{}-conv-{}-nodes-{}-dense{}".format(5, 128, 3, int(time.time()))
# print('Name:', NAME)
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.15, width_shift_range=0.1, height_shift_range=0.1)

# nets = 1
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
#     model[j].add(Dense(62, activation='softmax'))

#     model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#     print('Model', j+1, ':')
#     model[j].summary()

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# # TRAIN CNNs AND DISPLAY ACCURACIES
# epochs = 30
# history = [0] * nets
# results = [0] * nets
# for j in range(nets):
# #     X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
#     model[j].fit(X_train ,Y_train, batch_size=64,
#       epochs = epochs,
#       validation_split = 0.2, callbacks=[annealer, tensorboard])
# #     print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format
# #       (j+1,epochs,history[j].history['acc'][epochs-1],history[j].history['val_acc'][epochs-1]))
    
#     # PREDICT DIGITS FOR CNN J ON MNIST 10K TEST
#     results[j] = model[j].predict(X_test)
#     results2 = np.argmax(results[j],axis = 1)

#     # CALCULATE ACCURACY OF CNN J ON MNIST 10K TEST
#     c=0
#     for i in range(len(Y_test)):
#         if results2[i]!=y_test[i]:
#             c +=1
#     print("CNN %d: Test accuracy = %f" % ((float)(j+1,1-c/len(Y_test))))

# results2 = np.zeros( (X_test.shape[0],62) )
# for j in range(nets):
#     results2 = results2 + results[j]
# results2 = np.argmax(results2,axis = 1)
 
# # CALCULATE ACCURACY OF ENSEMBLE ON MNIST 10K TEST SET    
# c=0
# for i in range(len(Y_test)):
#     if results2[i]!=y_test[i]:
#         c +=1
# print("Ensemble Accuracy = %f" % ((float)(1-c/len(Y_test))))