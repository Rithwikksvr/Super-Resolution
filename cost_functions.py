import numpy as np
import pandas as pd
import os
import cv2
import shutil
import numpy
import math
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




def avg_psnr(X,y,model):
    s = 0
    pred1 = model.predict(X)
    val  = 0
    for i in range(len(pred1)):
        
        s = s + psnr(y[i]*255,pred1[i]*255)

    s = s/i
    return s

def mse_model(X,y,model):
	return np.mean(np.square(y - model.predict(X)))



