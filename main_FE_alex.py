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
dataset_dir = os.path.join(project_dir, 'preprocessing') # folder for the second challenge


#%% Image acquisition

# Getting paths to images
x_train = []
y_train = []
for image in os.listdir(os.path.join(dataset_dir, 'train')):
        if image.endswith(".jpg") and not image.startswith("."):
            pseudo_x = cv2.imread(os.path.join(dataset_dir, 'train', image))
            x_train.append(pseudo_x[:,:,0])
            y_train.append(image[0:3])
                

# Sparse implementation for dev speed
aux = 0
    
# Getting paths to images
x_val = []
y_val = []
for image in os.listdir(os.path.join(dataset_dir, 'val')):
        if image.endswith(".jpg") and not image.startswith("."):
            pseudo_x = cv2.imread(os.path.join(dataset_dir, 'val', image))
            x_val.append(pseudo_x[:,:,0])
            y_val.append(image[0:3])
                
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
x_train_arr = np.array(x_train)
x_val_arr = np.array(x_val)

del x_train
del x_val
gc.collect()

# LBP features extracted for 24 points and radius 8 from HSV images
# tst_lbp_vector_24_8 = lbp_process(x_train_arr, num_bins, 24, 8)
# val_lbp_vector_24_8 = lbp_process(x_val_arr, num_bins, 24, 8)

# LBP features extracted for 8 points and radius 1 from HSV images
trn_lbp_vector_8_1 = lbp_process(x_train_arr, num_bins, 8, 1)
val_lbp_vector_8_1 = lbp_process(x_val_arr, num_bins, 8, 1)


#%% Concatenate LBP & Mean Hue

# x_train = np.concatenate((mean_hue_train, trn_lbp_vector_8_1), axis=1)
# x_val = np.concatenate((mean_hue_val, val_lbp_vector_8_1), axis=1)

# save to csv file
np.savetxt('trn_lbp_vector_8_1.csv', trn_lbp_vector_8_1, delimiter=',')
np.savetxt('val_lbp_vector_8_1.csv', val_lbp_vector_8_1, delimiter=',')

# del train_lbp_vector_24_8
# del val_lbp_vector_24_8
del trn_lbp_vector_8_1
del val_lbp_vector_8_1
# del mean_hue_train
# del mean_hue_val


#%% Gabor
# Code for generating Gabor filters and deleting black filters
filters = build_filters()
filters = black_filters_delete(filters)

# Applying gabor filters to validation different configurations
val_imgs_filtered = filtered_image(x_val_arr, filters)
val_imgs_filtered = np.array(val_imgs_filtered)
del x_val_arr

# val_gabor_24_8 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 24, 8)
val_gabor_8_1 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 8, 1)
np.savetxt('val_gabor_8_1.csv', val_gabor_8_1, delimiter=',')

# Applying gabor filters to training different configurations

train_imgs_filtered = filtered_image(x_train_arr, filters)
# train_imgs_filtered = np.array(train_imgs_filtered)
# del x_train_arr

# train_gabor_24_8 = lbps_to_gaborIMG(train_imgs_filtered, num_bins, 24, 8)
train_gabor_8_1 = []
for img in train_imgs_filtered:
    with open("train_gabor_8_1.csv", "a") as f:
        np.savetxt(f, lbps_to_gaborIMG(img[np.newaxis], num_bins, 8, 1).flatten(), delimiter=',')
        f.write("\n")
    
del train_gabor_8_1, img, train_imgs_filtered, filters, f


#%%
# del train_imgs_filtered
# del x_train_24_8
# del x_train_8_1


#%% Concatenating Everything

x_train = np.concatenate((np.loadtxt('D:\\Uni\\Spain\\CADx\\csvfileeees\\mean_hsv_train.csv', dtype=np.float32, delimiter=',')[:,0][np.newaxis].T,
                          np.loadtxt('D:\\Uni\\Spain\\CADx\\csvfileeees\\trn_lbp_vector_8_1.csv', dtype=np.float32, delimiter=','),
                          np.loadtxt('D:\\Uni\\Spain\\CADx\\csvfileeees\\train_gabor_8_1.csv', dtype=np.float32, delimiter=',').reshape([4800,2048])
                          ), axis=1)
x_val = np.concatenate((np.loadtxt('D:\\Uni\\Spain\\CADx\\csvfileeees\\mean_hsv_val.csv', dtype=np.float32, delimiter=',')[0:1200,0][np.newaxis].T,
                          np.loadtxt('D:\\Uni\\Spain\\CADx\\csvfileeees\\val_lbp_vector_8_1.csv', dtype=np.float32, delimiter=','),
                          np.loadtxt('D:\\Uni\\Spain\\CADx\\csvfileeees\\val_gabor_8_1.csv', dtype=np.float32, delimiter=',')
                          ), axis=1)

y_train = np.array(y_train)
y_val = np.array(y_val)

# JUST IN CASE
gc.collect()


#%% Class imbalance handled
# import matplotlib.pyplot as plt
import pandas as pd
# Next to libraries are installed with the next command in anaconda
# conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

# autopct = "%.2f"

# sampling_strategy = "not minority"

# fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
# rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
# X_res, y_res = rus.fit_resample(x_train, y_train)
# y_res = pd.DataFrame(y_res)
# y_res.value_counts().plot.pie(autopct=autopct, ax=axs[0])
# axs[0].set_title("Under-sampling")

sampling_strategy = "not majority"
ros = RandomOverSampler(sampling_strategy=sampling_strategy)
X_res, y_res = ros.fit_resample(x_train, y_train)
y_res = pd.DataFrame(y_res)
# y_res.value_counts().plot.pie(autopct=autopct, ax=axs[1])
# axs[1].set_title("Over-sampling")

x_train = X_res
y_train = y_res.to_numpy()

del X_res, y_res

#%% Feature selection

param_kbest = SelectKBest(f_classif, k=1000)
param_kbest.fit(x_train, y_train)
# Saving
joblib.dump(param_kbest, 'feature_selector.sav') 
x_train_kbest = param_kbest.transform(x_train)  # Then we transform both the training an the test set
x_test_kbest = param_kbest.transform(x_val)

# In this challenge, feature normalization with a standard scaler is very useful
scaler = StandardScaler()
# Saving
joblib.dump(scaler, 'feature_scaler.sav') 

# The function fit_transform makes both fitting and transformation. It is equivalent to the function fit
# followed by the function transform
x_train = scaler.fit_transform(x_train_kbest)  # Again, the fitting applied only on the training set
x_val = scaler.transform(x_test_kbest)  # The test set instead is only transformed

# x_train = x_train_kbest
# x_val = x_test_kbest

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)


#%% KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
import joblib

param_grid_knn = {'n_neighbors': [3, 7, 15, 30]}

# Classifier
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv, refit=True)
grid_search_knn.fit(x_train, y_train)
params_best_knn = grid_search_knn.best_params_
print('Best parameters for kNN are = ', params_best_knn)

y_pred_knn = grid_search_knn.predict(x_val)

# accuracy on X_test
acc_knn = accuracy_score(y_val, y_pred_knn)
print("Accuracy kNN is = ", acc_knn)

# Saving
# joblib.dump(grid_search_knn, 'grid_search_knn.sav') 

# Loading
# loaded_knn = joblib.load('grid_search_knn.sav')
# result = loaded_knn.score(x_val, y_val)
# print (result)

#%% Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_val)

# accuracy on X_test
acc_NB = gnb.score(x_val, y_val)
print('Naive Bayes classifier: ', acc_NB)

# # Saving
# joblib.dump(gnb, 'grid_search_NB.sav') 

# # Loading
# loaded_NB = joblib.load('grid_search_NB.sav')
# result = loaded_NB.score(x_train, y_train)
# print (result)


#%% SVC classifier
from sklearn.svm import SVC

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                           'C': [0.01, 1, 3, 10, 100, 1000]},
                          {'kernel': ['poly'], 'C': [0.01, 3, 10, 100, 1000]},
                          {'kernel': ['sigmoid'], 'C': [0.01, 3, 10, 100, 1000]}
                          ]
svc = SVC()
clf = GridSearchCV(svc, parameters, cv=cv, refit=True)
clf.fit(x_train, y_train)

params_best_SVC = clf.best_params_
print('Best parameters for SVC with DD for are = ', params_best_SVC)

y_pred_svc = clf.predict(x_val)
 
# model accuracy for X_test 
acc_SVC = accuracy_score(y_val, y_pred_svc)
print("Accuracy SVC with DD is = ", acc_SVC)

# # Saving
# joblib.dump(clf, 'grid_search_SVC.sav') 

# # Loading
# loaded_SVC = joblib.load('grid_search_SVC.sav')
# result = loaded_SVC.score(x_val, y_val)
# print (result)

#%% Decission tree classifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
param_grid_dt = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2, 8, 10, 15, 20, 35, 50],
    'criterion': ['gini', 'entropy']
}

GS_dtc = GridSearchCV(estimator=dtc, param_grid=param_grid_dt, cv=cv, refit=True)
GS_dtc.fit(x_train, y_train)
params_best_dt = GS_dtc.best_params_
print('Best parameters for Decision Trees are = ', params_best_dt)

y_pred = GS_dtc.predict(x_val)
acc_DT = accuracy_score(y_val, y_pred)
print("Accuracy Random forest is = ", acc_DT)

# # Saving
# joblib.dump(GS_dtc, 'DecisionTreeClassifier.sav') 

# # Loading
# loaded_GS_dtc = joblib.load('DecisionTreeClassifier.sav')
# result = loaded_GS_dtc.score(x_val, y_val)
# print (result)


#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rfc = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [1000],
    'max_features': ['auto'],
    'max_depth': [50],
    'criterion': ['entropy']
}

GS_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=cv, refit=True)
GS_rfc.fit(x_train, y_train)
params_best_RF = GS_rfc.best_params_
print('Best parameters for Random forest are = ', params_best_RF)

y_pred = GS_rfc.predict(x_val)
acc_RF = accuracy_score(y_val, y_pred)
print("Accuracy Random forest is = ", acc_RF)

tst = confusion_matrix(y_val, y_pred)/y_val.shape[0]

# # Saving
# joblib.dump(GS_rfc, 'RandomForestClassifier.sav') 

# # Loading
# loaded_GS_rfc = joblib.load('RandomForestClassifier.sav')
# result = loaded_GS_rfc.score(x_val, y_val)
# print (result)

#%% Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier(random_state=42)
param_grid_et = {
    'n_estimators': [30, 100, 250, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2, 8, 10, 15, 20, 35, 50],
    'criterion': ['gini', 'entropy']
}

GS_etc = GridSearchCV(estimator=etc, param_grid=param_grid_et, cv=cv, refit=True)
GS_etc.fit(x_train, y_train)
params_best_ET = GS_etc.best_params_
print('Best parameters for Extra trees are = ', params_best_ET)

y_pred = GS_etc.predict(x_val)
acc_ET = accuracy_score(y_val, y_pred)
print("Accuracy Extra trees is = ", acc_ET)

# Saving
joblib.dump(GS_etc, 'ExtraTreesClassifier.sav') 

# # Loading
# loaded_GS_etc = joblib.load('ExtraTreesClassifier.sav')
# result = loaded_GS_etc.score(x_val, y_val)
# print (result)

#%% Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=29)
param_grid_gb = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [30, 100],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [3,  25, 50],
}

GS_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gb, cv=cv, refit=True)
GS_gbc.fit(x_train, y_train)
params_best_GB = GS_gbc.best_params_
print('Best parameters for Gradient boosting are = ', params_best_GB)

y_pred = GS_gbc.predict(x_val)
acc_GB = accuracy_score(y_val, y_pred)
print("Accuracy Gradient boosting is = ", acc_GB)

# # Saving
# joblib.dump(GS_gbc, 'GradientBoostingClassifier.sav') 

# # Loading
# loaded_GS_gbc = joblib.load('GradientBoostingClassifier.sav')
# result = loaded_GS_gbc.score(x_val, y_val)
# print (result)


