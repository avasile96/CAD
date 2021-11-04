# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:24:50 2021

@author: Manuel Ojeda & Alexandru Vasile
"""

# import tensorflow as tf
import os
import numpy as np
from skimage import io
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import cv2
import gc

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset2') # folder for the second challenge


# Sparse implementation for dev speed --> read every 10th image
aux = 0

# Getting paths to images
x_train = []
y_train = []
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                if aux % 10 == 0:
                    pseudo_x = cv2.imread(os.path.join(dataset_dir, 'train', case, image))
                    x_train.append(pseudo_x)
                    y_train.append(case)
                aux+=1

# Sparse implementation for dev speed
aux = 0
    
# Getting paths to images
x_val = []
y_val = []
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                if aux % 10 == 0:
                    pseudo_x = cv2.imread(os.path.join(dataset_dir, 'val', case, image))
                    x_val.append(pseudo_x)
                    y_val.append(case)
                aux+=1
                
del pseudo_x              
gc.collect()
# io.imshow(x_train[0])

#%% Preprocessing
from skimage.color import rgb2hsv

def hairRemoval(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter =(11, 11)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filter)
    black_hat = cv2.morphologyEx(otsu, cv2.MORPH_BLACKHAT,kernel)
    inpaint_img = cv2.inpaint(img, black_hat, 7, flags=cv2.INPAINT_TELEA)
    return inpaint_img


# List to array
x_train_arr = np.array(x_train)
x_val_arr = np.array(x_val)

### Color Space Transformation: RGB --> HSV ###
Y_val = []
Y_train = []

for i in range(0,x_train_arr.shape[0]):
    x_train_arr[i] = cv2.cvtColor(x_train_arr[i],cv2.COLOR_RGB2HSV)
    Y_train.append(y_train[i])
    
for i in range(0,x_val_arr.shape[0]):
    x_val_arr[i] = cv2.cvtColor(x_val_arr[i],cv2.COLOR_RGB2HSV)
    Y_val.append(y_val[i])
    
### Hair Removal
x_train_no_hair = np.zeros((x_train_arr.shape[0], x_train_arr.shape[1], x_train_arr.shape[2]))
x_val_no_hair = np.zeros((x_val_arr.shape[0], x_val_arr.shape[1], x_val_arr.shape[2]))

for i in range(x_train_no_hair.shape[0]):
    x_train_no_hair[i] = hairRemoval(x_train[i])

for i in range(x_val_no_hair.shape[0]):
    x_val_no_hair[i] = hairRemoval(x_val[i])

# tst = lbp_process(np.array(images_filtered), 256, 8, 1)
del y_train
del y_val
gc.collect()

#%% Feature extraction

mean_of_train = np.zeros(x_train_arr.shape[0])
mean_of_val = np.zeros(x_val_arr.shape[0])

# Mean of image
for i in range(x_train_arr.shape[0]):
    mean_of_train[i] = np.mean(x_train_arr[i,:,:,2])
mean_of_train = mean_of_train[np.newaxis].T
    
for i in range(x_val_arr.shape[0]):
    mean_of_val[i] = np.mean(x_val_arr[i,:,:,2])
mean_of_val = mean_of_val[np.newaxis].T
# SIFT

# Gabor filters
def build_filters():
    filters = []
    ksize = 9
    for theta in np.arange(0, np.pi, np.pi / 8):  # 8 ORIENTATIONS
        for lamda in np.arange(0, np.pi, np.pi / 4):  # 4 FREQUENCIES, 32 FILTERS IN TOTAL
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

def black_filters_delete(result):
    result_gabor = []
    for n in range(len(result)):
        if n % 4 != 0:
            result_gabor.append(result[n])
    return result_gabor

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(np.uint8(img), cv2.CV_8UC1, kern)
        np.maximum(accum, fimg, accum)
    return accum

def filtered_image(array, filters):
    # array_res = np.zeros_like(array)
    array_res = []
    for j in range(array.shape[0]):
        res = []
        for i in range(len(filters)):
            res1 = process(array[j], filters[i])
            res.append(np.asarray(res1))
        array_res.append(np.asarray(res))
    return array_res

# LBP
def lbp_process(array, bins, points, radius):
    histograms = np.zeros((array.shape[0], bins))
    for i in range(array.shape[0]):
        if len(array[i].shape)==3:
            img = cv2.cvtColor(array[i], cv2.COLOR_RGB2GRAY)
        else:
            img = array[i]
        lbp_result = local_binary_pattern(img, points, radius, method='ror')
        histogram_lbp, _ = np.histogram(lbp_result, bins=bins, density=True)
        histogram_lbp = histogram_lbp[np.newaxis]
        histograms[i,:] = histogram_lbp
    return histograms


# Code for generating Gabor filters and deleting black filters
filters = build_filters()
filters = black_filters_delete(filters)

# Applying gabor filters
images_filtered = filtered_image(x_train_no_hair[0:5,:,:], filters)
images_filtered = np.array(images_filtered)

# LBP features extracted for 24 points and radius 8 from HSV images
train_lbp_vector_24_8 = lbp_process(x_train_arr, 256, 24, 8)
val_lbp_vector_24_8 = lbp_process(x_val_arr, 256, 24, 8)

# LBP features extracted for 8 points and radius 1 from HSV images
train_lbp_vector_8_1 = lbp_process(x_train_arr, 256, 8, 1)
val_lbp_vector_8_1 = lbp_process(x_val_arr, 256, 8, 1)

# LBP features extracted for 24 points and radius 8 from images without hair
noHair_t_lbp_vector_24_8 = lbp_process(x_train_no_hair, 256, 24, 8)
noHair_v_lbp_vector_24_8 = lbp_process(x_val_no_hair, 256, 24, 8)

# LBP features extracted for 8 points and radius 1 from images without hair
noHair_t_lbp_vector_8_1 = lbp_process(x_train_no_hair, 256, 8, 1)
noHair_v_lbp_vector_8_1 = lbp_process(x_val_no_hair, 256, 8, 1)


#%% Inputs

x_train = np.concatenate((mean_of_train, train_lbp_vector_24_8, train_lbp_vector_8_1,
                          noHair_t_lbp_vector_24_8, noHair_t_lbp_vector_24_8), axis=1)
x_val = np.concatenate((mean_of_val, val_lbp_vector_24_8, val_lbp_vector_8_1,
                        noHair_v_lbp_vector_24_8, noHair_v_lbp_vector_24_8), axis=1)
y_train = np.array(Y_train)
y_val = np.array(Y_val)


#%% KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

# Classifier
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(x_train,y_train)

KNN_predict = neigh.predict(x_val)

# accuracy on X_test
acc_knn = neigh.score(x_val, y_val)
print('KNN classifier: ', acc_knn)

#%% Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_val)

# accuracy on X_test
accuracy = gnb.score(x_val, y_val)
print('Naive Bayes classifier: ', accuracy)

 
#%% SVC classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel = 'sigmoid', C = 1).fit(x_train, y_train)
svm_predictions = svm_model_linear.predict(x_val)
 
# model accuracy for X_test 
accuracy = svm_model_linear.score(x_val, y_val)
print('SVC classifier: ', accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(y_val, svm_predictions)


#%% Decission tree classifier
from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier()
dt = DecisionTreeClassifier(max_depth=2)  # This line works better than the previous one
dt.fit(x_train, y_train)

y_pred2 = dt.predict(x_val)
acc2 = accuracy_score(y_val, y_pred2)
print('Decission tree classifier: ', acc2)


#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(x_train, y_train)

RF_predict = rf.predict(x_val)
print('Random Forest classifier: ', accuracy_score(y_val, RF_predict))


#%% Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100, random_state=0)
et.fit(x_train, y_train)

ET_predict = et.predict(x_val)
print('Extra trees classifier: ', accuracy_score(y_val, ET_predict))


