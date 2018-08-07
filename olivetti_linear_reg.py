import numpy as np

import cv2


import numpy
import math
import os

import sklearn.linear_model
import sklearn.datasets

import matplotlib.pyplot as plt


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




data = sklearn.datasets.fetch_olivetti_faces()

imgs_hr = data['images']
imgs_lr = imgs_hr[:,::2,::2]

img_id = 1

sample_hr = imgs_hr[img_id]
sample_lr = imgs_lr[img_id]

#plt.imshow(sample_hr*255,cmap='gray')

y = imgs_hr.reshape((len(imgs_hr),64*64))
X = imgs_lr.reshape((len(imgs_lr),32*32))

model = sklearn.linear_model.LinearRegression()


model.fit(X,y)

pred = model.predict(X)

reshaped = pred.reshape((len(pred),64,64))


sample_res = reshaped[img_id]

plt.imshow(sample_res*255,cmap='gray')



