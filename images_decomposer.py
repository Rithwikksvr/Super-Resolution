import numpy as np

import cv2


import numpy
import math
import os

import sklearn.linear_model
import sklearn.datasets

import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import grad


plt.close('all')




folder = 'faces'

lst = os.listdir(folder)

imgs_hr = []
for i in lst:
    img = cv2.imread(folder+"/"+i,0)
    img = cv2.resize(img,(0,0), fx=0.25, fy=0.25)
    imgs_hr.append(img)
    
imgs_hr = np.array(imgs_hr)
imgs_lr = imgs_hr[:,::2,::2]

h,w = imgs_hr[0].shape



y = imgs_hr/255
X = imgs_lr/255

imgs_hr = None
imgs_lr = None

weights = np.random.random((int(h*h*.5 +w*w*.5),1))/10





def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def training_loss(weights):
    weights_1 = weights[:int(h*h*.5)].reshape((h,int(h/2)))
    weights_2 = weights[int(h*h*.5):].reshape((int(w/2),w))
    
    
    s = 0
    for i in range(len(X)):
        A = np.matmul(weights_1,X[i])
        B = np.matmul(A,weights_2)
        
        #print (y[i] - B)
        s = s + np.mean(np.square((y[i] - B)))
        #print (np.mean(np.square(y[i] - B)))
    return s/len(X)
    

print (training_loss(weights))



def pred(weights,img):
    weights_1 = weights[:int(h*h*.5)].reshape((h,int(h/2)))
    weights_2 = weights[int(h*h*.5):].reshape((int(w/2),w))
    
    
    temp = []
    
    if 1:
        A = np.matmul(weights_1,img)
        B = np.matmul(A,weights_2)
        
        #print (y[i] - B)
        
        #print (np.mean(np.square(y[i] - B)))
    return B
    
    
training_gradient = grad(training_loss)
alpha = .001
loss=training_loss(weights)
for i in range(10000):
    
    weights-= training_gradient(weights)*alpha
    if loss-training_loss(weights)<0.003:
        break
    loss=training_loss(weights)
    print (i,loss)
    
    

    
    
print (psnr(pred(weights,X[0])*255,y[0]*255))

ans = pred(weights,X[0])


plt.imshow(ans,cmap='gray')

