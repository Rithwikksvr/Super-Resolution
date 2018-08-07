# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:36:37 2018

@author: Rithwik
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
A = np.genfromtxt('arrays/Epoch_136000_A.out', delimiter=",")
B = np.genfromtxt('arrays/Epoch_136000_B.out', delimiter=",")


img = cv2.imread('faces/image_0001.jpg',0)

img = cv2.resize(img,(0,0), fx=0.25, fy=0.25)

img = img[::2,::2]
x1 = np.matmul(A,img)
x2 = np.matmul(x1,B)



plt.imshow(x2,cmap='gray')






