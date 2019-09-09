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


file = "../image.png"

img = cv2.imread(file, cv2.IMREAD_COLOR)
img28 = image_resize(img, width=28)
img28_gray = cv2.cvtColor(img28, cv2.COLOR_BGR2GRAY)
(thresh, img28_bw) = cv2.threshold(img28_gray, 150, 255, cv2.THRESH_BINARY)
# (thresh, img28_bw) = cv2.threshold(img28_gray, 180, 255, cv2.THRESH_BINARY)
img28_gray_inv = cv2.bitwise_not(img28_bw)
cv2.imshow('image', img28_gray_inv)
cv2.waitKey(2000)
img28_gray_inv = array(img28_gray_inv) / 255
# prediction_set = zeros((1, 28, 28))
# prediction_set[0] = img28_gray_inv / 255
prediction_set = reshape(img28_gray_inv, (1, 28, 28, 1))

# model = tf.keras.models.load_model("../model5epochs.hdf5")
model = tf.keras.models.load_model("../finalModel.hdf5")
# model = tf.keras.models.load_model("../finalModel4.hdf5")
print(model.predict_classes(prediction_set))
print(model.predict(prediction_set))