# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:24:50 2021

@author: vasil
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
import gc
import matplotlib as plt

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pickle import dump
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import Model, Sequential

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# https://www.tensorflow.org/tutorials/keras/keras_tuner

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 

# Generator Parameters
img_size = (450, 600, 3) # RGB imges!
# img_size = (225, 300,3) # RGB imges!
batch_size = 8

# Getting paths to images
train_img_paths = []
img_label = []
aux = 0
# pseudo_x = np.zeros()
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                if (aux%10==0):
                    train_img_paths.append(os.path.join(dataset_dir, 'train', case, image)) # for DL
                    img_label.append(case)
                aux+=1
    
val_img_paths = []
val_label = []
aux = 0
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                if (aux%10==0):
                    val_img_paths.append(os.path.join(dataset_dir, 'val', case, image)) # for DL
                    val_label.append(case)
                aux+=1

     
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
            # img = io.imread(path, as_gray = False)
            img = io.imread(path, as_gray = False)
            # img = resize(img, (img.shape[0] // 2, img.shape[1] // 2),
                       # anti_aliasing=True)
            x[j] = img
            
            if path[-10:-8] == 'ls':
                y[j] = 1
            else:
                y[j] = 0
                
        return x, tf.keras.utils.to_categorical(y, num_classes=2)

traingen = SkinImageDatabase(batch_size, img_size, train_img_paths, img_label)
valgen = SkinImageDatabase(batch_size, img_size, val_img_paths, val_label)

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
# model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
# Show a summary of the model. Check the number of trainable parameters
model.summary()

# tf.config.run_functions_eagerly(False) # result won't be affected by eager/graph mode
# tf.data.experimental.enable_debug_mode()

model.compile(loss=tf.losses.CategoricalCrossentropy(from_logits = True),
              optimizer='adam',
              metrics=['acc'])

#%%
checkpoint_filepath = os.path.join(source_dir, 'model.h5') # DEFINE NAME OF MODEL
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

#%%
# Train the model
history = model.fit(
                    x=traingen, 
                    batch_size=batch_size, 
                    epochs=10, 
                    verbose='auto',
                    callbacks=model_checkpoint_callback,
                    validation_data=valgen, 
                    shuffle=True,
                    )


predictions = model.predict(valgen.__getitem__(0)[0])



#%%
from sklearn.metrics import confusion_matrix
n_batches = len(valgen)

confusion_matrix(
    np.concatenate([np.argmax(valgen[i][1], axis=1) for i in range(n_batches)]),    
    np.argmax(model.predict(valgen, steps=n_batches), axis=1) 
)

#%%
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig(os.path.join('drive/MyDrive/performance_charts/basic_' + name_arch + '_model', 'accuracy_plot.png'))
plt.show()
































