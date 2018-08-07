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

def cost(A,B,X,y):
    pred = np.matmul(np.matmul(A,X),B)
    return np.mean(np.square(pred - y))



def mse(img1,img2):
    return np.mean(np.square(img1 - img2))


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




    


folder = 'faces'
lst = os.listdir(folder)
hr_img = cv2.resize(cv2.imread('faces/'+lst[0],0),(0,0),fx=0.5,fy=.5)/255
lr_img = hr_img[::2,::2]
(h,w) = hr_img.shape


A = np.random.random((h,h//2))/10
B = np.random.random((w//2,w))/10


A_copy = np.copy(A)
B_copy = np.copy(B)
        


cost(A,B,lr_img,hr_img)


trainer = grad(cost,argnum=[0,1])
alpha = .01
iterations = 100000


for i in range(iterations):
    grad_A,grad_B = trainer(A,B,lr_img,hr_img)
    A -=alpha*grad_A
    B -=alpha*grad_B
    
    
    if i%1000==0:
        if np.isnan(A.sum())==True or np.isnan(B.sum())==True:
            A = np.copy(A_copy)
            B = np.copy(B_copy)
            alpha = alpha/10
            
    if i%10000==0:
        alpha = alpha*10
        A_copy = np.copy(A)
        B_copy = np.copy(B)
        
   
    if i%100==0:
        ans = np.matmul(np.matmul(A,lr_img),B)
        
        print (i,mse(ans,hr_img),psnr(ans*255,hr_img*255))
        
        
    if i%8000==0:
        np.savetxt('arrays2/Epoch_'+str(i)+'_A.out', A, delimiter=',')
        np.savetxt('arrays2/Epoch_'+str(i)+'_B.out', B, delimiter=',')
        
        
    
    
    