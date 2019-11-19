import cv2
import tensorflow as tf
from numpy import zeros, array, reshape

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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


file = "../sample.jpg"

img = cv2.imread(file, cv2.IMREAD_COLOR)
img28 = image_resize(img, width=64)
img28_gray = cv2.cvtColor(img28, cv2.COLOR_BGR2GRAY)
(thresh, img28_bw) = cv2.threshold(img28_gray, 180, 255, cv2.THRESH_BINARY)
img28_gray_inv = cv2.bitwise_not(img28_gray)
print(img28_gray_inv.shape)
cv2.imshow('image', img28_gray_inv)
cv2.waitKey(2000)
img28_gray_inv = cv2.transpose(img28_gray_inv)
cv2.imshow('image', img28_gray_inv)
cv2.waitKey(2000)
prediction_set = reshape(img28_gray_inv, (-1, 28, 28, 1))
prediction_set = prediction_set.astype('float16')
print(prediction_set[0].shape)
model = tf.keras.models.load_model("logs/modelsBalancedComplete/checkpoint.hdf5")
print(model.predict_classes(prediction_set))
print(model.predict(prediction_set))