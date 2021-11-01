# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:24:50 2021

@author: Manuel Ojeda & Alexandru Vasile
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import gc

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset2') # folder for the second challenge

# Generator Parameters
img_size = (450, 600,3) # RGB imges!
batch_size = 32

# Getting paths to images
train_img_paths = []

# Getting paths to images
x_train = []
y_train = []
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                pseudo_x = io.imread(os.path.join(dataset_dir, 'train', case, image), as_gray = False)
                x_train.append(pseudo_x)
                y_train.append(case)
    
# Getting paths to images
x_val = []
y_val = []
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                # val_img_paths.append(os.path.join(dataset_dir, 'val', case, image))
                pseudo_val = io.imread(os.path.join(dataset_dir, 'val', case, image))
                x_val.append(pseudo_val)
                y_val.append(case)

pseudo_x=None                
pseudo_val=None
gc.collect()
# io.imshow(x_train[0])

#%% Preprocessing
from skimage.color import rgb2hsv

# List to array
# x_train_arr = np.array(x_train)
# x_val_arr = np.array(x_val)

# Section to select different images of different range, to select from 3 classes
# DON'T FORGET TO DELETE WHEN THE PROJECT IT'S DONE
x_train_arr = x_train[100:300] + x_train[900:1100] + x_train[1800:1900]
x_train_arr = np.array(x_train_arr)
x_val_arr = np.array(x_val)

y_train = y_train[100:300] + y_train[900:1100] + y_train[1800:1900]


x_train = None
x_val = None
gc.collect()


# Color Space Transformation
m = 0
x_train_hsv = np.zeros([500,450,600,3]) # instead of the 2000 images, because of memory
x_val_hsv = np.zeros([500,450,600,3])

for i in range(1,502,10):
    if i!=1:
        x_train_hsv[m:i-1,:,:,:] = rgb2hsv(x_train_arr[m:i-1,:,:,:])
        x_val_hsv[m:i-1,:,:,:] = rgb2hsv(x_val_arr[m:i-1,:,:,:])
        
        m = i-1
        
    # every 100 images, we collect the garbage (RAM saving)
    if i % 100 == 0:
        gc.collect()


#%% Feature extraction

feature_vector_train = np.zeros(x_train_hsv.shape[0])
feature_vector_val = np.zeros(x_val_hsv.shape[0])

# Mean of image
for i in range(1,x_train_hsv.shape[0]):
    feature_vector_train[i] = np.mean(x_train_hsv[i,:,:,2])
    feature_vector_val[i] = np.mean(x_val_hsv[i,:,:,2])
    
# SIFT

# LBP


#%% Inputs


x_train = feature_vector_train[np.newaxis].T
x_val = feature_vector_val[np.newaxis].T
y_train = np.array(y_train)
y_val = np.array(y_val)
# np.where(y_train == 'bcc',1,0)

#%% KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

# Classifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x_train,y_train)

KNN_predict = neigh.predict(x_val)

# accuracy on X_test
acc_knn = neigh.score(x_val, y_val)
print(acc_knn)

#%% Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_val)

# accuracy on X_test
accuracy = gnb.score(x_val, y_val)
print(accuracy)

 
#%% SVM classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel = 'linear', C = 3).fit(x_train, y_train)
svm_predictions = svm_model_linear.predict(x_val)
 
# model accuracy for X_test 
accuracy = svm_model_linear.score(x_val, y_val)
print(accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(y_val, svm_predictions)


#%% Decission tree classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred2 = dt.predict(x_val)
acc2 = accuracy_score(y_val, y_pred2)
print(acc2)


#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(x_train, y_train)

RF_predict = rf.predict(x_val)
print(accuracy_score(y_val, RF_predict))


#%% Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100, random_state=0)
et.fit(x_train, y_train)

ET_predict = et.predict(x_val)
print(accuracy_score(y_val, ET_predict))


