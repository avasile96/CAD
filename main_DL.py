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

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pickle import dump
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential

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

aux = 0
# pseudo_x = np.zeros()
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                if (aux%10==0):
                    train_img_paths.append(os.path.join(dataset_dir, 'train', case, image)) # for DL
                    # pseudo_x = io.imread(os.path.join(dataset_dir, 'train', case, image), as_gray = False)
                    # x.append(pseudo_x)
                    img_label.append(case)
                aux+=1
    
val_img_paths = []
aux = 0
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                if (aux%10==0):
                    val_img_paths.append(os.path.join(dataset_dir, case, image))
                aux+=1

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


x = SkinImageDatabase(batch_size, img_size, train_img_paths, img_label)

#%% Architecture

vgg19 = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=img_size,
        pooling=None,
        classifier_activation="softmax",
        )

# Freeze all the layers
for layer in vgg19.layers[:]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in vgg19.layers:
    print(layer, layer.trainable)
    
# Create the model
model = Sequential()
# Add the vgg convolutional base model
model.add(vgg19)
# Add new layers
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax')) #TODO
# Show a summary of the model. Check the number of trainable parameters
model.summary()

predictions = vgg19.predict(x)







































