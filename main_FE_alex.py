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

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'preprocessing2') # folder for the second challenge


#%% Image acquisition

# Getting paths to images
x_train = []
y_train = []
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                pseudo_x = cv2.imread(os.path.join(dataset_dir, 'train', case, image))
                x_train.append(pseudo_x)
                y_train.append(case)
                

# Sparse implementation for dev speed
aux = 0
    
# Getting paths to images
x_val = []
y_val = []
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                pseudo_x = cv2.imread(os.path.join(dataset_dir, 'val', case, image))
                x_val.append(pseudo_x)
                y_val.append(case)
                
                
gc.collect()

#%% Preprocessing

# List to array
x_train_arr = np.array(x_train)
x_val_arr = np.array(x_val)

mean_of_train_hue = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_train_sat = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_train_val = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T

mean_of_val_hue = np.zeros(x_val_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_val_sat = np.zeros(x_val_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_val_val = np.zeros(x_val_arr.shape[0], dtype = np.float32)[np.newaxis].T

### Color Space Transformation: RGB --> HSV ###
Y_val = []
Y_train = []

for i in range(0,x_train_arr.shape[0]):
    x_train_arr[i] = cv2.cvtColor(x_train_arr[i],cv2.COLOR_RGB2HSV)
    x_train_arr[i] = preprocessing(x_train_arr[i])
    Y_train.append(y_train[i])
    
    # filename_train = 'F:\\DISCO_DURO\\Mixto\\Subjects\\GitHub\\preprocessing2\\train\\{}_{}.jpg'.format(y_train[i],i)
    # cv2.imwrite(filename_train,x_train_arr[i,:,:,2])
    
    mean_of_train_hue[i] = np.mean(x_train_arr[i,:,:,0]) # getting the mean of the hue channel
    mean_of_train_sat[i] = np.mean(x_train_arr[i,:,:,1]) # getting the mean of the sat channel
    mean_of_train_val[i] = np.mean(x_train_arr[i,:,:,2]) # getting the mean of the val channel
    # cv2.imshow("segim",x_train_arr[i])
    # cv2.waitKey(5000)
    
    
for i in range(0,x_val_arr.shape[0]):
    x_val_arr[i] = cv2.cvtColor(x_val_arr[i],cv2.COLOR_RGB2HSV)
    x_val_arr[i] = preprocessing(x_val_arr[i])
    Y_val.append(y_val[i])
    
    # filename_val = 'F:\\DISCO_DURO\\Mixto\\Subjects\\GitHub\\preprocessing2\\val\\{}_{}.jpg'.format(y_val[i],i)
    # cv2.imwrite(filename_val,x_val_arr[i,:,:,2])
    
    mean_of_val_hue[i] = np.mean(x_val_arr[i,:,:,0]) # getting the mean of the hue channel
    mean_of_val_sat[i] = np.mean(x_val_arr[i,:,:,1]) # getting the mean of the sat channel
    mean_of_val_val[i] = np.mean(x_val_arr[i,:,:,2]) # getting the mean of the val channel


mean_of_test = np.concatenate((mean_of_train_hue, mean_of_train_sat, mean_of_train_val), axis=1)
mean_of_val = np.concatenate((mean_of_val_hue, mean_of_val_sat, mean_of_val_val), axis=1)
# save to csv file
# path_mean_of_test = os.path.join(dataset_dir, 'mean_hsv_test.csv')
# np.savetxt(path_mean_of_test, mean_of_test, delimiter=',')
# np.savetxt('mean_hsv_val.csv', mean_of_val, delimiter=',')

# # Test reading
# # load_hsv_train = np.loadtxt('mean_hsv_train.csv', dtype=np.float32, delimiter=',')
# # load_hsv_val = np.loadtxt('mean_hsv_val.csv', dtype=np.float32, delimiter=',')

#%% Hair Removal

# noHair_train_path = os.path.join(dataset_dir, 'noHair_test')
# preprocess_path = os.path.join(dataset_dir, 'preprocess_test')

# def hairRemoval(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     filter =(11, 11)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filter)
#     black_hat = cv2.morphologyEx(otsu, cv2.MORPH_BLACKHAT,kernel)
#     inpaint_img = cv2.inpaint(img, black_hat, 7, flags=cv2.INPAINT_TELEA)
#     return inpaint_img


# for i in range(x_train_arr.shape[0]):
#     img_rgb = cv2.cvtColor(x_train_arr[i], cv2.COLOR_HSV2BGR)
#     img = hairRemoval(img_rgb)
    
#     filename_noHair_test = os.path.join(noHair_train_path, str(i) + '.jpg')
#     cv2.imwrite(filename_noHair_test, img)
    
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#     filename_preprocess_test = os.path.join(preprocess_path, str(i) + '.jpg')
#     cv2.imwrite(filename_preprocess_test, img_gray)


# gc.collect()

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


#%% Only LBPs different configurations

# List to array
x_train_arr = np.array(x_train)
x_val_arr = np.array(x_val)


# LBP features extracted for 24 points and radius 8 from HSV images
tst_lbp_vector_24_8 = lbp_process(x_train_arr, num_bins, 24, 8)
val_lbp_vector_24_8 = lbp_process(x_val_arr, num_bins, 24, 8)

# LBP features extracted for 8 points and radius 1 from HSV images
tst_lbp_vector_8_1 = lbp_process(x_train_arr, num_bins, 8, 1)
val_lbp_vector_8_1 = lbp_process(x_val_arr, num_bins, 8, 1)


#%% Concatenate different vectors, if needed

# x_train = np.concatenate((mean_hue_train, train_lbp_vector_24_8, train_lbp_vector_8_1), axis=1)
# x_val = np.concatenate((mean_hue_val, val_lbp_vector_24_8, val_lbp_vector_8_1), axis=1)

# # save to csv file
# np.savetxt('train_mean_lbp248_lbp81.csv', x_train, delimiter=',')
# np.savetxt('val_mean_lbp248_lbp81.csv', x_val, delimiter=',')

# del train_lbp_vector_24_8
# del val_lbp_vector_24_8
# del train_lbp_vector_8_1
# del val_lbp_vector_8_1
# del mean_hue_train
# del mean_hue_val


#%%

# Code for generating Gabor filters and deleting black filters
filters = build_filters()
filters = black_filters_delete(filters)


# Applying gabor filters to validation different configurations
val_imgs_filtered = filtered_image(x_val_arr, filters)
val_imgs_filtered = np.array(val_imgs_filtered)

val_gabor_24_8 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 24, 8)
val_gabor_8_1 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 8, 1)


# Applying gabor filters to training different configurations
train_imgs_filtered = filtered_image(x_train_arr, filters)
train_imgs_filtered = np.array(train_imgs_filtered)

train_gabor_24_8 = lbps_to_gaborIMG(train_imgs_filtered, num_bins, 24, 8)
train_gabor_8_1 = lbps_to_gaborIMG(train_imgs_filtered, num_bins, 8, 1)


#%%
# del train_imgs_filtered
# del x_train_24_8
# del x_train_8_1


#%%

# # save to csv file
# np.savetxt('noHair_val_gb8_lbp248_lbp81.csv', x_noHair_val_concat, delimiter=',')

gc.collect()


#%%

# load_hsv_train = np.loadtxt('noHair_train_gb8_lbp248_lbp81.csv', dtype=np.float32, delimiter=',')
# load_hsv_val = np.loadtxt('noHair_val_gb8_lbp248_lbp81.csv', dtype=np.float32, delimiter=',')

# load_1st_train = np.loadtxt('train_mean_lbp248_lbp81.csv', dtype=np.float32, delimiter=',')
# load_1st_val = np.loadtxt('val_mean_lbp248_lbp81.csv', dtype=np.float32, delimiter=',')


#%% Inputs

# x_tst = np.concatenate((tst_gabor_24_8, tst_gabor_8_1), axis=1)
# x_val = np.concatenate((load_1st_val, load_hsv_val, x_noHair_val_concat, 
                        # x_noHair_val_concat_og), axis=1)

# x_train = np.concatenate((load_1st_train,
#                           x_noHair_train_concat_og), axis=1)
# x_val = np.concatenate((load_1st_val, 
#                         x_noHair_val_concat_og), axis=1)

y_train = np.array(y_train)
y_val = np.array(y_val)

# JUST IN CASE
gc.collect()



#%% Feature selection

param_kbest = SelectKBest(f_classif, k=2000)
param_kbest.fit(x_train, y_train)
x_train_kbest = param_kbest.transform(x_train)  # Then we transform both the training an the test set
x_test_kbest = param_kbest.transform(x_val)

# In this challenge, feature normalization with a standard scaler is very useful
scaler = StandardScaler()

# The function fit_transform makes both fitting and transformation. It is equivalent to the function fit
# followed by the function transform
x_train = scaler.fit_transform(x_train_kbest)  # Again, the fitting applied only on the training set
x_val = scaler.transform(x_test_kbest)  # The test set instead is only transformed

# x_train = x_train_kbest
# x_val = x_test_kbest

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)


#%% KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

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


#%% Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_val)

# accuracy on X_test
acc_NB = gnb.score(x_val, y_val)
print('Naive Bayes classifier: ', acc_NB)


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
print('Best parameters for Random forest are = ', params_best_dt)

y_pred = GS_dtc.predict(x_val)
acc_DT = accuracy_score(y_val, y_pred)
print("Accuracy Random forest is = ", acc_DT)


#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [30, 100, 250, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2, 8, 10, 15, 20, 35, 50],
    'criterion': ['gini', 'entropy']
}

GS_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=cv, refit=True)
GS_rfc.fit(x_train, y_train)
params_best_RF = GS_rfc.best_params_
print('Best parameters for Random forest are = ', params_best_RF)

y_pred = GS_rfc.predict(x_val)
acc_RF = accuracy_score(y_val, y_pred)
print("Accuracy Random forest is = ", acc_RF)


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


#%% Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=29)
param_grid_gb = {
    'criterion': ['deviance', 'exponential'],
    'learning_rate': [0.1, 0.01, 0.001, 0.3, 0.003, 0.003],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [3, 8, 10, 15, 20, 35, 50],
}

GS_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gb, cv=cv, refit=True)
GS_gbc.fit(x_train, y_train)
params_best_GB = GS_gbc.best_params_
print('Best parameters for Gradient boosting are = ', params_best_GB)

y_pred = GS_gbc.predict(x_val)
acc_GB = accuracy_score(y_val, y_pred)
print("Accuracy Gradient boosting is = ", acc_GB)


