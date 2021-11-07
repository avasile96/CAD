# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 05:55:10 2021

@author: 52331
"""

import os
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Directories
source_dir = os.getcwd() # current working directory
csvFiles_dir = os.path.join(source_dir, 'CSV_files')

#%% Loading CSV files

labels_train_path = os.path.join(csvFiles_dir, 'y_labels_train_pre.csv')
labels_val_path = os.path.join(csvFiles_dir, 'y_labels_val_pre.csv')

# Variables to get the labels for train and validation
y_train = np.loadtxt(labels_train_path, dtype=(str), delimiter=',')
y_val = np.loadtxt(labels_val_path, dtype=(str), delimiter=',')

mean_train_path = os.path.join(csvFiles_dir, 'train_mean_lbp248,lbp81.csv')
mean_val_path = os.path.join(csvFiles_dir, 'val_mean_lbp248,lbp81.csv')

mean_hue_train = np.loadtxt(mean_train_path, dtype=np.float32, delimiter=',')
mean_hue_val = np.loadtxt(mean_val_path, dtype=np.float32, delimiter=',')

gb_lbp_train_path = os.path.join(csvFiles_dir, 'train_gb8_lbp248_lbp81.csv')
gb_lbp_val_path = os.path.join(csvFiles_dir, 'val_gb8_lbp248_lbp81.csv')

gb_lbp_train = np.loadtxt(gb_lbp_train_path, dtype=np.float32, delimiter=',')
gb_lbp_val = np.loadtxt(gb_lbp_val_path, dtype=np.float32, delimiter=',')

nH_gb_lbp_train_path = os.path.join(csvFiles_dir, 'noHair_train_gb8_lbp248_lbp81.csv')
nH_gb_lbp_val_path = os.path.join(csvFiles_dir, 'noHair_val_gb8_lbp248_lbp81.csv')

nH_gb_lbp_train = np.loadtxt(nH_gb_lbp_train_path, dtype=np.float32, delimiter=',')
nH_gb_lbp_val = np.loadtxt(nH_gb_lbp_val_path, dtype=np.float32, delimiter=',')

nH_lbp_train_path = os.path.join(csvFiles_dir, 'noHair_train_lbp248_lbp81.csv')
nH_lbp_val_path = os.path.join(csvFiles_dir, 'noHair_val_lbp248_lbp81.csv')

nH_lbp_train = np.loadtxt(nH_lbp_train_path, dtype=np.float32, delimiter=',')
nH_lbp_val = np.loadtxt(nH_lbp_val_path, dtype=np.float32, delimiter=',')

# Concatenate all the files to get x_train and x_val
x_train = np.concatenate((mean_hue_train, gb_lbp_train, nH_gb_lbp_train,
                          nH_lbp_train), axis=1)
x_val = np.concatenate((mean_hue_val, gb_lbp_val, nH_gb_lbp_val,
                        nH_lbp_val), axis=1)

# save to csv file, WHEN FINAL FILE IS READY
# np.savetxt(csvfile_train_path, x_train, delimiter=',')
# np.savetxt(csvfile_val_path, x_val, delimiter=',')


#%% Delete of unused variables

del mean_hue_train
del gb_lbp_train
del nH_gb_lbp_train
del mean_hue_val
del gb_lbp_val
del nH_gb_lbp_val
del nH_lbp_train
del nH_lbp_val

#%% Class imbalance handled
import matplotlib.pyplot as plt
import pandas as pd
# Next to libraries are installed with the next command in anaconda
# conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

autopct = "%.2f"

sampling_strategy = "not minority"

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_res, y_res = rus.fit_resample(x_train, y_train)
y_res = pd.DataFrame(y_res)
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)


#%% KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

param_grid_knn = {'n_neighbors': [3, 7, 15, 30]}

# Classifier
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv, refit=True)
grid_search_knn.fit(x_train, y_train)
params_best = grid_search_knn.best_params_
print('Best parameters for kNN are = ', params_best)

y_pred_knn = grid_search_knn.predict(x_val)

# accuracy on X_test
print("Accuracy kNN is = ", accuracy_score(y_val, y_pred_knn))


#%% Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_val)

# accuracy on X_test
accuracy = gnb.score(x_val, y_val)
print('Naive Bayes classifier: ', accuracy)


#%% SVC classifier
from sklearn.svm import SVC

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                           'C': [0.01, 1, 3, 10, 100, 1000]},
                          {'kernel': ['linear'], 'C': [0.01, 3, 10, 100, 1000]},
                          {'kernel': ['poly'], 'C': [0.01, 3, 10, 100, 1000]},
                          {'kernel': ['sigmoid'], 'C': [0.01, 3, 10, 100, 1000]}
                          ]
svc = SVC()
clf = GridSearchCV(svc, parameters, cv=cv, refit=True)
clf.fit(x_train, y_train)

params_best = clf.best_params_
print('Best parameters for SVC with DD for are = ', params_best)

y_pred_svc = clf.predict(x_val)
 
# model accuracy for X_test 
print("Accuracy SVC with DD is = ", accuracy_score(y_val, y_pred_svc))


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
params_best = GS_dtc.best_params_
print('Best parameters for Random forest are = ', params_best)

y_pred = GS_dtc.predict(x_val)
print("Accuracy Random forest is = ", accuracy_score(y_val, y_pred))


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
params_best = GS_rfc.best_params_
print('Best parameters for Random forest are = ', params_best)

y_pred = GS_rfc.predict(x_val)
print("Accuracy Random forest is = ", accuracy_score(y_val, y_pred))


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
params_best = GS_etc.best_params_
print('Best parameters for Extra trees are = ', params_best)

y_pred = GS_etc.predict(x_val)
print("Accuracy Extra trees is = ", accuracy_score(y_val, y_pred))
