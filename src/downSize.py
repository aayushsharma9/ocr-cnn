import cv2
from numpy import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def classTextToInt(row_label):
    switcher = {
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '0': 10,
        'a': 37,
        'b': 38,
        'c': 13,
        'd': 39,
        'e': 40,
        'f': 41,
        'g': 42,
        'h': 43,
        'i': 19,
        'j': 20,
        'k': 21,
        'l': 22,
        'm': 23,
        'n': 44,
        'o': 25,
        'p': 26,
        'q': 45,
        'r': 46,
        's': 29,
        't': 46,
        'u': 31,
        'v': 32,
        'w': 33,
        'x': 34,
        'y': 35,
        'z': 36,
        'A': 11,
        'B': 12,
        'C': 13,
        'D': 14,
        'E': 15,
        'F': 16,
        'G': 17,
        'H': 18,
        'I': 19,
        'J': 20,
        'K': 21,
        'L': 22,
        'M': 23,
        'N': 24,
        'O': 25,
        'P': 26,
        'Q': 27,
        'R': 28,
        'S': 29,
        'T': 30,
        'U': 31,
        'V': 32,
        'W': 33,
        'X': 34,
        'Y': 35,
        'Z': 36,
        'Random': 47
    }
    return switcher.get(row_label, None)

folderToFind = "by_class/"
offsetLength = len(folderToFind)

def imageResize (image, width = None, height = None, inter = cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def imageGray (image):
    imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imageG

def downScaleImage(path, index):
    image = cv2.imread(path)   
    grayImage = imageGray(image)
    thresh = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 40)
    resizedImage = imageResize(thresh, height = 64)
    arr = array(resizedImage)
    return arr

def runForCSV(inputCSV, outputCSV, outputLabel):
    labelArray = []
    startTime = time.time()
    firstIterationStart = 0.0
    firstIterationEnd = 0.0
    df1 = pd.read_csv(inputCSV)
    with open(outputCSV, 'a') as f:
        for index, row in df1.iterrows():
            if index == 1:
                firstIterationStart = time.time()
            print('Progress:',(index/len(df1))*100, '%                              ', end='\r')
            temp = downScaleImage(row['fileName'], index)
            filePath = row['fileName']
            fileClass = filePath[filePath.find(folderToFind) + offsetLength]
            toAdd = temp.ravel().astype(np.uint8)
            dft = pd.DataFrame([toAdd])
            labelArray.append(classTextToInt(fileClass))
            dft.to_csv(f, index=False, header=False)
            if index == 1:
                firstIterationEnd = time.time()
                print("Estimated time:", ((firstIterationEnd - firstIterationStart) * len(df1)) / 60, "minutes")
    dft = dft = pd.DataFrame(labelArray)
    dft.to_csv(outputLabel, index=False)
    print("Done in", ((time.time() - startTime) / 60), "minutes")

runForCSV('trainingRecords.csv', 'trainingDataset.csv', 'trainingLabels.csv')
runForCSV('validationRecords.csv', 'validationDataset.csv', 'validationLabels.csv')

# dataArray = []
# labelArray = []
# df1 = pd.read_csv('trainingRecords.csv')
# for index, row in df1.iterrows():
#     if index % 1000 == 0:
#         print('Progress:',(index/len(df1))*100, '%                              ', end='\r')
#     temp = downScaleImage(row['fileName'], index)
#     filePath = row['fileName']
#     fileClass = filePath[filePath.find(folderToFind) + offsetLength]
#     toAdd = temp.ravel()
#     dataArray.append(toAdd)
#     labelArray.append(classTextToInt(fileClass))

# print('read and convert train done')

# dft = pd.DataFrame(dataArray)
# dft.to_csv('trainingDataset.csv', index=False)

# print('save train data done')

# dft = dft = pd.DataFrame(labelArray)
# dft.to_csv('trainingLabels.csv', index=False)

# print('save train labels done')

# dataArray = []
# labelArray = []
# df1 = pd.read_csv('validationRecords.csv')
# for index, row in df1.iterrows():
#     if index % 1000 == 0:
#         print('Progress:',(index/len(df1))*100, '%                              ', end='\r')
#     temp = downScaleImage(row['fileName'], index)
#     filePath = row['fileName']
#     fileClass = filePath[filePath.find(folderToFind) + offsetLength]
#     toAdd = temp.ravel() / 255
#     toAdd = np.insert(toAdd, 0, classTextToInt(fileClass))
#     dataArray.append(toAdd)
#     labelArray.append(classTextToInt(fileClass))

# print('read and convert validate done')

# dft = pd.DataFrame(dataArray)
# dft.to_csv('validationDataset.csv', index=False)

# print('save validate data done')

# dft = dft = pd.DataFrame(labelArray)
# dft.to_csv('validationLabels.csv', index=False)

# print('save validate labels done')
