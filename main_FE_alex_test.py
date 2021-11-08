# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:24:50 2021

@author: Manuel Ojeda & Alexandru Vasile
"""

# import tensorflow as tf
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import cv2
import gc
from libb import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'preprocessing_test') # folder for the second challenge


#%% Image acquisition

# Getting paths to images
x_train = []
y_train = []
for image in os.listdir(os.path.join(dataset_dir)):
    if image.endswith(".jpg") and not image.startswith("."):
        pseudo_x = cv2.imread(os.path.join(dataset_dir, image))
        x_train.append(pseudo_x[:,:,0])
        y_train.append(image[0:6])
                
del pseudo_x
#%% Loading CSV files

# # Test reading for csv files with mean values of HSV images
# load_hsv_train = np.loadtxt('mean_hsv_train.csv', dtype=np.float32, delimiter=',')
# load_hsv_val = np.loadtxt('mean_hsv_val.csv', dtype=np.float32, delimiter=',')

# mean_hue_train = load_hsv_train[:,0][np.newaxis].T
# mean_hue_val = load_hsv_val[:,0][np.newaxis].T

# del load_hsv_train
# del load_hsv_val

#%% Functions
# Gabor filters
def build_filters():
    filters = []
    ksize = 9
    for theta in np.arange(0, np.pi, np.pi / 8):  # 8 ORIENTATIONS
        for lamda in np.arange(0, np.pi, np.pi / 2):  # 4 FREQUENCIES, 32 FILTERS IN TOTAL
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

def black_filters_delete(result):
    result_gabor = []
    for n in range(len(result)):
        if n % 2 != 0:
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
            res.append(np.asarray(res1, dtype=np.float16))
        array_res.append(np.asarray(res, dtype=np.float16))
    return array_res

def lbps_to_gaborIMG(array, bins, points, radius):
    x_temp = np.zeros((array.shape[0], array.shape[1]*bins))
    for m in range(array.shape[0]):
        lbp_gabor = lbp_process(array[m], bins, points, radius)
        for n in range(lbp_gabor.shape[0]):
            x_temp[m,n*bins:n*bins+bins] = lbp_gabor[n]
    return x_temp

# LBP
def lbp_process(array, bins, points, radius):
    histograms = np.zeros((array.shape[0], bins))
    for i in range(array.shape[0]):
        if len(array[i].shape)==3:
            img = cv2.cvtColor(np.float32(array[i]), cv2.COLOR_RGB2GRAY)
        else:
            img = array[i]
        lbp_result = local_binary_pattern(img, points, radius, method='ror')
        histogram_lbp, _ = np.histogram(lbp_result, bins=bins)
        histogram_lbp = histogram_lbp[np.newaxis]
        histograms[i,:] = histogram_lbp
    return histograms

num_bins = 256


#%% Get LBP

# List to array
x_test_arr = np.array(x_train)

del x_train
gc.collect()

# LBP features extracted for 24 points and radius 8 from HSV images
# tst_lbp_vector_24_8 = lbp_process(x_train_arr, num_bins, 24, 8)
# val_lbp_vector_24_8 = lbp_process(x_val_arr, num_bins, 24, 8)

# LBP features extracted for 8 points and radius 1 from HSV images
tst_lbp_vector_8_1 = lbp_process(x_test_arr, num_bins, 8, 1)


#%% Concatenate LBP & Mean Hue

# x_train = np.concatenate((mean_hue_train, trn_lbp_vector_8_1), axis=1)
# x_val = np.concatenate((mean_hue_val, val_lbp_vector_8_1), axis=1)

# save to csv file
np.savetxt('tst_lbp_vector_8_1.csv', tst_lbp_vector_8_1, delimiter=',')

# del train_lbp_vector_24_8
# del val_lbp_vector_24_8
del tst_lbp_vector_8_1
# del mean_hue_train
# del mean_hue_val


#%% Gabor
# Code for generating Gabor filters and deleting black filters
filters = build_filters()
filters = black_filters_delete(filters)

# Applying gabor filters to test different configurations
tst_imgs_filtered = filtered_image(x_test_arr, filters)
tst_imgs_filtered = np.array(tst_imgs_filtered)
del x_test_arr

tst_gabor_8_1 = lbps_to_gaborIMG(tst_imgs_filtered, num_bins, 8, 1)
np.savetxt('tst_gabor_8_1.csv', tst_gabor_8_1, delimiter=',')


#%%
# del tst_imgs_filtered
# del x_train_24_8
# del x_train_8_1


#%% Concatenating Everything

x_test = np.concatenate((np.loadtxt('mean_test_hsv.csv', dtype=np.float32, delimiter=',')[:,0][np.newaxis].T,
                          np.loadtxt('tst_lbp_vector_8_1.csv', dtype=np.float32, delimiter=','),
                          np.loadtxt('tst_gabor_8_1.csv', dtype=np.float32, delimiter=',')
                          ), axis=1)

# JUST IN CASE
gc.collect()


#%% Feature selection

#Load feature selection model (fsm)
loaded_fsm = joblib.load('feature_selector.sav')

x_test_kbest = loaded_fsm.transform(x_test)  # Then we transform both the training an the test set

# In this challenge, feature normalization with a standard scaler is very useful
scaler = joblib.load('feature_scaler.sav')

# The function fit_transform makes both fitting and transformation. It is equivalent to the function fit
# followed by the function transform
x_test = scaler.fit_transform(x_test_kbest)



#%% KNN Classifier

loaded_knn = joblib.load('grid_search_knn.sav')
result = loaded_knn.predict(x_test)
print (result)

#%% Naive Bayes classifier

loaded_NB = joblib.load('grid_search_NB.sav')
result = loaded_NB.predict(x_test)
print (result)

#%% SVC classifier

loaded_SVC = joblib.load('grid_search_SVC.sav')
result = loaded_SVC.predict(x_test)
print (result)

#%% Decission tree classifier

loaded_GS_dtc = joblib.load('DecisionTreeClassifier.sav')
result = loaded_GS_dtc.predict(x_test)
print (result)

#%% Random Forest ################## WINNER #########################
loaded_GS_rfc = joblib.load('RandomForestClassifier.sav')
result = loaded_GS_rfc.predict(x_test)
print (result)

#%% Getting the final CSV

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['nv_', 'les'])

result_numberical = le.transform(result) # les = 0, nv = 1
result_numberical = np.where(result_numberical == 1,0,1)  # les = 1, nv = 0
csv_content = np.concatenate(y_train,result, axis=1)

np.savetxt('tst_gabor_8_1.csv', tst_gabor_8_1, delimiter=',')

#%% Extra trees classifier

loaded_GS_etc = joblib.load('ExtraTreesClassifier.sav')
result = loaded_GS_etc.predict(x_test)
print (result)


#%% Gradient Boosting Classifier

loaded_GS_gbc = joblib.load('GradientBoostingClassifier.sav')
result = loaded_GS_gbc.predict(x_test)
print (result)

