import cv2
import tensorflow as tf
import numpy as np
from class_names import classes

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


img = cv2.imread("../sample_text.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
cv2.imshow('img', img)
cv2.waitKey(1000)
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
cv2.imshow("contours", img)
d = 0

model = tf.keras.models.load_model("logs/modelsBalancedFinal1/model1.hdf5")

for ctr in contours:
    x, y, w, h = cv2.boundingRect(ctr)
    roi = img[y:y + h, x:x + w]
    print(roi.shape)
    roi = image_resize(roi, width=None, height=28)
    prediction_set = np.reshape(roi, (-1, 28, 28, 1))
    prediction_set = prediction_set.astype('float16')
    print(classes[model.predict_classes(prediction_set)])
    cv2.imshow('character: %d' % d, roi)
    cv2.imwrite('character_%d.png' % d, roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    d += 1
