import numpy as np

import cv2


import numpy
import math
import os

import sklearn.linear_model
import sklearn.datasets

import matplotlib.pyplot as plt

plt.close('all')

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

test_error = []

x_axis = []



for k in range(50):
    
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
    
    
    
    y = imgs_hr.reshape((len(imgs_hr),w*h))/255
    X = imgs_lr.reshape((len(imgs_lr),int(h*w*.25)))/255
    
    imgs_hr = None
    imgs_lr = None
    
    
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=k)
    
    print ("started training")
    
    model.fit(X,y)
    
    
    
    Train_error = np.sum(np.square(y - model.predict(X)))/len(y)
    
    print ("Training  Error with k = "+ str(k)+"  is ",Train_error)
    
    s = 0
    pred1 = model.predict(X)
    
    for i in range(len(pred1)):
        s = s + psnr(y[i]*255,pred1[i]*255)
    s = s/i
    
    print ("Average training PSNR : with K = "+str( k) + ' is '+str( s))
    
    
    #######Test Data####
    
    
    folder = 'test'
    
    lst = os.listdir(folder)
    
    imgs_hr = []
    for i in lst:
        img = cv2.imread(folder+"/"+i,0)
        img = cv2.resize(img,(0,0), fx=0.25, fy=0.25)
        imgs_hr.append(img)
        
    imgs_hr = np.array(imgs_hr)
    imgs_lr = imgs_hr[:,::2,::2]
    
    
    y = imgs_hr.reshape((len(imgs_hr),w*h))/255
    X = imgs_lr.reshape((len(imgs_lr),int(h*w*.25)))/255
    
    
    imgs_hr=None
    imgs_lr = None
    
    
    img_id=0
    
    
    pred = model.predict(X[:,:])
    
    
    test = np.mean(np.square(y - model.predict(X)))
    
    print ("Testing Error with k = "+ str(k)+"  is ",test)
    
    
    
    s = 0
    pred1 = model.predict(X)
    
    for i in range(len(pred1)):
        s = s + psnr(y[i]*255,pred1[i]*255)
    s = s/len(y)
    
    print ("Average Testing PSNR : with K = "+ str(k) + ' is '+ str(s))
    
    






    
    
    
    reshaped = pred.reshape((len(pred),h,w))
    
    
    
    y = y.reshape((len(X),h,w))
    
    figure = plt.figure()
    
    plt.imshow(reshaped[img_id],cmap='gray')
    
    figure = plt.figure()
    
    plt.imshow(y[img_id],cmap='gray')
    
    model = None
    neight = None
    
    test_error.append(test)
    
    x_axis.append(k)
    
    
    

