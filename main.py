# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:34:02 2021

@author: usuari
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier

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


# Inputs
x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# FOR DL
# train_gen = SkinImageDatabase(batch_size, img_size, train_img_paths, case)
# val_gen = SkinImageDatabase(batch_size, img_size, val_img_paths, case)

# Classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x,y)
    
    
# Predictions
print(neigh.predict([[1.1]])) # hard classification prediction
print(neigh.predict_proba([[0.9]])) # confidence score prediction
