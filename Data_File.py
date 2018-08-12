import numpy as np
import pandas as pd
import os
import cv2
import shutil

def data_writer(folder,train_prop,test_prop,v_prop):

	lst = os.listdir(folder)
	
	lst.sort()
	train_examples = int(len(lst)*train_prop)
	train_imgs = lst[:train_examples]
	test_examples = int(len(lst)*test_prop)
	test_imgs = lst[train_examples:train_examples+test_examples]
	v_examples = int(len(lst)*v_prop)
	v_imgs = lst[train_examples+test_examples:train_examples+test_examples+v_examples]

	
	name_dir = 'faces'
	empty_dir(name_dir)
	for i in train_imgs:

		shutil.copy(folder+"/"+ i,name_dir+"/"+i)


	
	name_dir = 'test'
	empty_dir(name_dir)
	for i in test_imgs:

		shutil.copy(folder+"/"+ i,name_dir+"/"+i)

	name_dir = 'cross_val'
	empty_dir(name_dir)

	for i in v_imgs:

		shutil.copy(folder+"/"+ i,name_dir+"/"+i)


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



def empty_dir(folder):
	lst = os.listdir(folder)
	for i in lst:
		os.remove(folder + "/" + i)
