# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:33:42 2021

@author: 52331
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import gc

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pickle import dump
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import Model, Sequential

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random

from keras.callbacks import Callback
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import datetime



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# https://www.tensorflow.org/tutorials/keras/keras_tuner

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 

# Directores to save the results for binary and multiclass
classification_type = 'binary' # 'binary' or 'multi'
DL_folder = os.path.join(source_dir, 'DL_folders')
csvs_path = os.path.join(DL_folder, classification_type, 'CSVs')
models_path = os.path.join(DL_folder, classification_type, 'models')
plots_path = os.path.join(DL_folder, classification_type, 'plots')


# Generator Parameters
img_size = (450, 600, 3) # RGB imges!
batch_size = 8

# Getting paths to images
train_img_paths = []
img_label = []
aux = 0
# pseudo_x = np.zeros()
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                # if (aux%10==0):
                train_img_paths.append(os.path.join(dataset_dir, 'train', case, image)) # for DL
                img_label.append(case)
                # aux+=1
    
val_img_paths = []
val_label = []
aux = 0
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                # if (aux%10==0):
                val_img_paths.append(os.path.join(dataset_dir, 'val', case, image)) # for DL
                val_label.append(case)
                # aux+=1

     
gc.collect()
    
#%% Generator
class SkinImageDatabase(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, img_label):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        random.shuffle(self.input_img_paths)
        self.img_label = img_label
        self.shuffle = True

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        y = np.zeros((self.batch_size,) + (1,), dtype='uint8')
        for j, path in enumerate(batch_input_img_paths):
            
            img = io.imread(path, as_gray = False)
            
            x[j] = img
            
            if path[-10:-8] == 'ls':
                y[j] = 1
            else:
                y[j] = 0
                
        return x, tf.keras.utils.to_categorical(y, num_classes=2)

traingen = SkinImageDatabase(batch_size, img_size, train_img_paths, img_label)
valgen = SkinImageDatabase(batch_size, img_size, val_img_paths, val_label)

#%%
model0 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\basic_DenseNet121_model_val_acc_81.h5') 
model1 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\dropout_5_VGG16_model_val_acc_84.h5') 
models = [model0, model1]

#%% Prediction
# make predictions
yhats = [model.predict(valgen) for model in models]
yhats = np.array(yhats)

# sum across ensembles
summed = np.sum(yhats, axis=0)
# argmax across classes
outcomes = np.argmax(summed, axis=1)

#%%
n_batches = len(valgen)

true_labels = np.concatenate([np.argmax(valgen[i][1], axis=1) for i in range(n_batches)])

print(accuracy_score(true_labels, outcomes))

