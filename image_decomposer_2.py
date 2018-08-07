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





folder = 'single'
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




A = np.genfromtxt('arrays/Epoch_136000_A.out', delimiter=",")
B = np.genfromtxt('arrays/Epoch_136000_B.out', delimiter=",")

A = np.random.random((A.shape))
B = np.random.random((B.shape))


def cost1(A):
    s = 0
    for i in range(len(X)):
        x1 = np.matmul(A,X[i])
        x2 = np.matmul(x1,B)
        s = s + np.mean(np.square((y[i] - x2)))
    return s/len(X)


def cost2(B):
    s = 0
    for i in range(len(X)):
        x1 = np.matmul(A,X[i])
        x2 = np.matmul(x1,B)
        s = s + np.mean(np.square((y[i] - x2)))
    return s/len(X)



def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




def pred(img):
    x1 = np.matmul(A,img)
    x2 = np.matmul(x1,B)
    return x2
    


grad_A = grad(cost1)
grad_B = grad(cost2)

alpha = .001


A_copy = np.copy(A)
B_copy = np.copy(B)
        
for i in range(1000000):
    
      
    if i%1000==0:
        if np.isnan(A.sum())==True or np.isnan(B.sum())==True:
            A = np.copy(A_copy)
            B = np.copy(B_copy)
            alpha = alpha/10
            
    if i%10000==0:
        alpha = alpha*10
        A_copy = np.copy(A)
        B_copy = np.copy(B)
        
    A -= grad_A(A)*alpha
    B -= grad_B(B)*alpha
    
    
    loss=cost1(A)
    
    if i%100==0:
        
        print (i,loss,psnr(pred(X[0])*255,y[0]*255))
        
        
    if i%8000==0:
        np.savetxt('arrays/Epoch_'+str(i)+'_A.out', A, delimiter=',')
        np.savetxt('arrays/Epoch_'+str(i)+'_B.out', B, delimiter=',')
    


    
    
    
    
print (psnr(pred(X[0])*255,y[0]*255))

ans = pred(X[0])


plt.imshow(ans,cmap='gray')

