# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:24:50 2021

@author: vasil
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import gc

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 

# Generator Parameters
img_size = (450, 600,3) # RGB imges!
batch_size = 32

# Getting paths to images
train_img_paths = []
img_label = []

x = []
# pseudo_x = np.zeros()
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                # train_img_paths.append(os.path.join(dataset_dir, 'train', case, image)) # for DL
                pseudo_x = io.imread(os.path.join(dataset_dir, 'train', case, image), as_gray = False)
                x.append(pseudo_x)
                img_label.append(case)
    
val_img_paths = []    
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                val_img_paths.append(os.path.join(dataset_dir, case, image))

pseudo_x=None                
gc.collect()
    
#%% Generator
class SkinImageDatabase(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, img_label):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.img_label = img_label

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_labels = self.img_label[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = False)
            x[j] = img

        y = np.array((self.batch_size,) + tuple(self.img_label), dtype="str")
        for j, path in enumerate(batch_target_img_labels):
            y[j] = path
            
        return x, y

#%% Preprocessing
from skimage.color import rgb2hsv

# List to array
x_arr = np.array(x)
x = None
gc.collect()

# Color Space Transformation
m = 0
x_hsv = np.zeros([60,450,600,3])
for i in range(1,62,10):
    if i!=1:
        x_hsv[m:i-1,:,:,:] = rgb2hsv(x_arr[m:i-1,:,:,:])
        print(i-1)
        m = i-1
    if i % 100 == 0:
        gc.collect()


#%% Feature extraction

feature_vector = np.zeros(x_hsv.shape[0])

# Mean of image
for i in range(1,x_hsv.shape[0]):
    feature_vector[i] = np.mean(x_hsv[i,:,:,2])
    
# SIFT

# LBP


#%% Inputs


x_train = feature_vector[np.newaxis].T
y_train = np.array(img_label,dtype='U')
np.where(y_train == 'les',1,0)

# Classifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x_train,y_train[0:60])
    
    
# Predictions
print(neigh.predict([[1]])) # hard classification prediction
print(neigh.predict_proba([[0.9]])) # confidence score prediction