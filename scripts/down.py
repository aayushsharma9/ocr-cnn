import cv2
from numpy import array
import pandas as pd
import numpy as np
import time

def classTextToInt(row_label):
    switcher = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'a': 10,
        'b': 11,
        'c': 12,
        'd': 13,
        'e': 14,
        'f': 15,
        'g': 16,
        'h': 17,
        'i': 18,
        'j': 19,
        'k': 20,
        'l': 21,
        'm': 22,
        'n': 23,
        'o': 24,
        'p': 25,
        'q': 26,
        'r': 27,
        's': 28,
        't': 29,
        'u': 30,
        'v': 31,
        'w': 32,
        'x': 33,
        'y': 34,
        'z': 35,
        'A': 36,
        'B': 37,
        'C': 38,
        'D': 39,
        'E': 40,
        'F': 41,
        'G': 42,
        'H': 43,
        'I': 44,
        'J': 45,
        'K': 46,
        'L': 47,
        'M': 48,
        'N': 49,
        'O': 50,
        'P': 51,
        'Q': 52,
        'R': 53,
        'S': 54,
        'T': 55,
        'U': 56,
        'V': 57,
        'W': 58,
        'X': 59,
        'Y': 60,
        'Z': 61
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
    resizedImage = imageResize(grayImage, height = 28)
    returnedImage = cv2.bitwise_not(resizedImage)
    arr = array(returnedImage)
    return arr

def runForCSV(inputCSV, outputCSV):
    startTime = time.time()
    firstIterationStart = 0.0
    firstIterationEnd = 0.0
    df1 = pd.read_csv(inputCSV)
    with open(outputCSV, 'a') as f:
        for index, row in df1.iterrows():
            if index == 1:
                firstIterationStart = time.time()
            print('Progress:',(index/len(df1))*100, '%                              ', end='\r')
            temp = downScaleImage(row['filename'], index)
            filePath = row['filename']
            fileClass = filePath[filePath.find(folderToFind) + offsetLength]
            toAdd = temp.ravel() / 255
            toAdd = np.insert(toAdd, 0, classTextToInt(fileClass))
            dft = pd.DataFrame([toAdd])
            dft.to_csv(f, index=False, header=False)
            if index == 1:
                firstIterationEnd = time.time()
                print("Estimated time:", ((firstIterationEnd - firstIterationStart) * len(df1)) / 60, "minutes")
    pd.read_csv(outputCSV)
    print("Done in", ((time.time() - startTime) / 60), "minutes")



runForCSV('trainingRecords.csv', 'data/trainingRecords.csv')
runForCSV('validationRecords.csv', 'data/traivalidationRecordsningRecords.csv')


# df1 = pd.read_csv('trainingRecords.csv')
# temp = downScaleImage(df1.loc[0, 'filename'], 0)
# filePath = df1.loc[0, 'filename']
# print(filePath)
# fileClass = filePath[filePath.find(folderToFind) + offsetLength]
# toAdd = temp.ravel() / 255
# toAdd = np.insert(toAdd, 0, classTextToInt(fileClass))
# dataArray.append(toAdd)
# print(fileClass)
# print(dataArray)
