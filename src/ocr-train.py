import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import switcher
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger
from sklearn.model_selection import train_test_split

def classTextToInt(row_label):
    return switcher.data.get(row_label, None)

folderToFind = "by_class\\"
offsetLength = len(folderToFind)


def imageResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def imageGray(image):
    imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imageG


def downScaleImage(path, index):
    image = cv2.imread(path)
    grayImage = imageGray(image)
    thresh = cv2.adaptiveThreshold(
        grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 40)
    resizedImage = imageResize(thresh, height=128)
    arr = np.array(resizedImage)
    return arr


SAVE_PATH = "logs\\modelsBalancedComplete"
num_classes = 48

X_train = []
Y_train = []
df1 = pd.read_csv('trainingRecords.csv')
for index, row in df1.iterrows():
    if index % 1000 == 0:
        print('Progress:', (index/len(df1))*100,
              '%                              ', end='\r')
    temp = downScaleImage(row['fileName'], index)
    filePath = row['fileName']
    fileClass = filePath[filePath.find(folderToFind) + offsetLength]
    toAdd = temp.ravel()
    X_train.append(toAdd)
    Y_train.append(classTextToInt(fileClass))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = np.reshape(X_train, (-1, 128, 128, 1))

X_test = []
Y_test = []
df1 = pd.read_csv('validationRecords.csv')
for index, row in df1.iterrows():
    if index % 1000 == 0:
        print('Progress:', (index/len(df1))*100,
              '%                              ', end='\r')
    temp = downScaleImage(row['fileName'], index)
    filePath = row['fileName']
    fileClass = filePath[filePath.find(folderToFind) + offsetLength]
    toAdd = temp.ravel()
    X_test.append(toAdd)
    Y_test.append(classTextToInt(fileClass))

X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = np.reshape(X_test, (-1, 128, 128, 1))

datagen = ImageDataGenerator(
    rotation_range=15, zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1)

nets = 1
model = [0] * nets
tensorboard = [0] * nets
csv_logger = [0] * nets
modelCheckpoint = [0] * nets


for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=3, activation='relu',
                        input_shape=(128, 128, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=5, strides=2,
                        padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=5, strides=2,
                        padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size=4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(num_classes, activation='softmax'))

    model[j].compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[
                     'accuracy', 'sparse_categorical_accuracy', 'mean_squared_error', 'mean_absolute_error'])

    NAME = "{}-conv-{}-nodes-{}-dense{}".format(7, 48, 1, int(time.time()))

    tensorboard[j] = TensorBoard(
        log_dir='{}\\{}______{}'.format(SAVE_PATH, NAME, j))

    csv_logger[j] = CSVLogger(
        '{}\\csvlogs\\{}__{}.csv'.format(SAVE_PATH, NAME, j), separator=',', append=False)

    modelCheckpoint[j] = ModelCheckpoint(
        'checkpoint.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    print('Model', j+1, ':')
    model[j].summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
epochs = 30
history = [0] * nets

for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
        X_train, Y_train, test_size=0.2)
    history[j] = model[j].fit_generator(datagen.flow(X_train2, Y_train2, batch_size=64),
                                        epochs=epochs, steps_per_epoch=X_train2.shape[0]/64,
                                        validation_data=(X_val2, Y_val2),
                                        callbacks=[annealer, tensorboard[j], csv_logger[j], modelCheckpoint[j]])
    model[j].save("{}\\model{0:d}.hdf5".format(SAVE_PATH, j+1))


for j in range(nets):
    model[j].evaluate(X_test, Y_test)
