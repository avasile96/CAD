# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:24:50 2021

@author: Manuel Ojeda & Alexandru Vasile
"""

# import tensorflow as tf
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import cv2
import gc
from libb import preprocessing

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'preprocessing') # folder for the second challenge


#%% Reading data

# Training Labels
y_train_pre = []

for image in os.listdir(os.path.join(dataset_dir, 'train')):
    if image.endswith(".jpg") and not image.startswith("."):
        y_train_pre.append(image[0:3])

# Validation labels
y_val_pre = []

for image in os.listdir(os.path.join(dataset_dir, 'val')):
    if image.endswith(".jpg") and not image.startswith("."):
        y_val_pre.append(image[0:3])
                
        
y_val_pre = np.array(y_val_pre)
path = os.path.join(dataset_dir, 'y_labels_val_pre.csv')
np.savetxt(path, y_val_pre, delimiter=',', fmt='%s')              
gc.collect()



# mean_of_test = np.concatenate((mean_of_train_hue, mean_of_train_sat, mean_of_train_val), axis=1)


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


#%% Only LBPs 

# List to array
# x_train_arr = np.array(x_train)
# x_val_arr = np.array(x_val)


# LBP features extracted for 24 points and radius 8 from HSV images
# tst_lbp_vector_24_8 = lbp_process(x_train_arr, num_bins, 24, 8)
# val_lbp_vector_24_8 = lbp_process(x_val_arr, num_bins, 24, 8)

# LBP features extracted for 8 points and radius 1 from HSV images
# tst_lbp_vector_8_1 = lbp_process(x_train_arr, num_bins, 8, 1)
# val_lbp_vector_8_1 = lbp_process(x_val_arr, num_bins, 8, 1)


#%% 

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
x_noHair_train_concat = np.zeros((2000,4096))
aux = 0
for image in os.listdir(os.path.join(dataset_dir, 'train_noHair')):
    img = cv2.imread(os.path.join(dataset_dir, 'train_noHair', image))
    img = img[np.newaxis]
    # Applying gabor filters to training set 24 points, 8 radius
    train_imgs_filtered = filtered_image(img, filters)
    train_imgs_filtered = np.array(train_imgs_filtered)
    x_train_24_8 = lbps_to_gaborIMG(train_imgs_filtered, num_bins, 24, 8)
    
    # Applying gabor filters to training set 8 points, 1 radius
    train_imgs_filtered = filtered_image(img, filters)
    train_imgs_filtered = np.array(train_imgs_filtered)
    x_train_8_1 = lbps_to_gaborIMG(train_imgs_filtered, num_bins, 8, 1)
    
    temp = np.concatenate((x_train_24_8, x_train_8_1), axis=1)
    x_noHair_train_concat[aux] = temp
    aux+=1

#%%

# Code for generating Gabor filters and deleting black filters
filters = build_filters()
filters = black_filters_delete(filters)
# Applying gabor filters to validation set 24 points, 8 radius
# val_imgs_filtered = filtered_image(x_train_arr, filters)
# val_imgs_filtered = np.array(val_imgs_filtered)
# tst_gabor_24_8 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 24, 8)

# tst_gabor_8_1 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 8, 1)

#%%
# del train_imgs_filtered
# del x_train_24_8
# del x_train_8_1

tst_noHair_val_concat = np.zeros((226,4096), dtype=np.float32)
aux = 0
for image in os.listdir(os.path.join(dataset_dir, 'noHair_test')):
    img = cv2.imread(os.path.join(dataset_dir, 'noHair_test', image))
    img = img[np.newaxis]
    # Applying gabor filters to validation set 24 points, 8 radius
    val_imgs_filtered = filtered_image(img, filters)
    val_imgs_filtered = np.array(val_imgs_filtered)
    tst_24_8 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 24, 8)
    
    # Applying gabor filters to training set 8 points, 1 radius
    tst_8_1 = lbps_to_gaborIMG(val_imgs_filtered, num_bins, 8, 1)
    
    temp = np.concatenate((tst_24_8, tst_8_1), axis=1)
    tst_noHair_val_concat[aux] = temp
    aux+=1

#%%

# # save to csv file
# np.savetxt('noHair_val_gb8_lbp248_lbp81.csv', x_noHair_val_concat, delimiter=',')

# JUST IN CASE
del val_imgs_filtered
# del x_val_24_8
# del x_val_8_1

gc.collect()


#%%

x_noHair_test_concat_og = np.zeros((226,512), dtype=np.float32)
# x_noHair_val_concat_og = np.zeros((500,512), dtype=np.float32)
aux = 0
for image in os.listdir(os.path.join(dataset_dir, 'noHair_test')):
    img = cv2.imread(os.path.join(dataset_dir, 'noHair_test', image))
    img = img[np.newaxis]
    # LBP features extracted for 24 points and radius 8 from images without hair
    noHair_t_lbp_vector_24_8 = lbp_process(img, num_bins, 24, 8)
    # noHair_v_lbp_vector_24_8 = lbp_process(img, num_bins, 24, 8)
    
    # LBP features extracted for 8 points and radius 1 from images without hair
    noHair_t_lbp_vector_8_1 = lbp_process(img, num_bins, 8, 1)
    # noHair_v_lbp_vector_8_1 = lbp_process(img, num_bins, 8, 1)
    
    # temp_val = np.concatenate((noHair_v_lbp_vector_24_8, noHair_v_lbp_vector_8_1), axis=1)
    temp_train = np.concatenate((noHair_t_lbp_vector_24_8, noHair_t_lbp_vector_8_1), axis=1)
    # x_noHair_val_concat_og[aux] = temp_val
    x_noHair_test_concat_og[aux] = temp_train
    aux+=1



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
# y_val = np.array(y_val)

# JUST IN CASE
gc.collect()
# del noHair_t_lbp_vector_24_8
# del noHair_t_lbp_vector_8_1

path_test = os.path.join(dataset_dir, 'noHair_t_lbp248_lbp81.csv')
np.savetxt(path_test, x_noHair_test_concat_og, delimiter=',')


#%% Feature selection
param_kbest = SelectKBest(f_classif, k=1000)
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

#%% KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

# Classifier
neigh = KNeighborsClassifier(n_neighbors=12)
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

svm_model_linear = SVC(kernel = 'rbf', C = 3).fit(x_train, y_train)
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

rf = RandomForestClassifier(max_depth=30, random_state=0)
rf.fit(x_train, y_train)

RF_predict = rf.predict(x_val)
print('Random Forest classifier: ', accuracy_score(y_val, RF_predict))


#%% Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100, random_state=0)
et.fit(x_train, y_train)

ET_predict = et.predict(x_val)
print('Extra trees classifier: ', accuracy_score(y_val, ET_predict))


