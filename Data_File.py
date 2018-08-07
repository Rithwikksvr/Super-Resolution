import numpy as np
import pandas as pd
import os
import cv2


def Data(folder):

	
	lst = os.listdir(folder)
	imgs_hr = []
	for i in lst:
	    img = cv2.imread(folder+"/"+i,0)
	    img = cv2.resize(img,(0,0), fx=0.25, fy=0.25)
	    imgs_hr.append(img)
	imgs_hr = np.array(imgs_hr)
	imgs_hr = imgs_hr.astype(float)
	imgs_lr = imgs_hr[:,::2,::2]
	h,w = imgs_hr[0].shape
	y = imgs_hr/255.0
	X = imgs_lr/255.0
	imgs_hr = None
	imgs_lr = None

	return X,y
