#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 03:48:16 2021

@author: manuel
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
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



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# https://www.tensorflow.org/tutorials/keras/keras_tuner

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 

# Directores to save the results for binary and multiclass
classification_type = 'multi' # 'binary' or 'multi'
DL_folder = os.path.join(source_dir, 'DL_folders')
csvs_path = os.path.join(DL_folder, classification_type, 'CSVs')
models_path = os.path.join(DL_folder, classification_type, 'models')
plots_path = os.path.join(DL_folder, classification_type, 'plots')

# Generator Parameters
img_size = (450, 600, 3) # RGB imges!
batch_size = 8
classes_num = 3


train_img_paths = []
img_label = []
aux = 0
# pseudo_x = np.zeros()
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                #if (aux%10==0):
                train_img_paths.append(os.path.join(dataset_dir, 'train', case, image)) # for DL
                img_label.append(case)
                #aux+=1
    
val_img_paths = []
val_label = []
aux = 0
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                #if (aux%10==0):
                val_img_paths.append(os.path.join(dataset_dir, 'val', case, image)) # for DL
                val_label.append(case)
                #aux+=1
                
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
            
            if path[-11:-8] == 'bcc':
              y[j] = 0
            elif path[-11:-8] == 'bkl':
              y[j] = 1
            else:
              y[j] = 2               

                
        return x, tf.keras.utils.to_categorical(y, num_classes=classes_num)

traingen = SkinImageDatabase(batch_size, img_size, train_img_paths, img_label)
valgen = SkinImageDatabase(batch_size, img_size, val_img_paths, val_label)


#%% Architecture

base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=img_size,
        pooling=None
        )

# Freeze all the layers
for layer in base_model.layers[:]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)
    
# Create the model
model = Sequential()
# Add the vgg convolutional base model
model.add(base_model)
# Add new layers
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(classes_num, activation='softmax'))
# Show a summary of the model. Check the number of trainable parameters
model.summary()

# To get the number of layers the model contains and use it when the model is saved
num_layers = 0
for layer in model.layers:
    num_layers+=1

base_model_name = model.get_layer(index=0).name

# tf.config.run_functions_eagerly(False) # result won't be affected by eager/graph mode
# tf.data.experimental.enable_debug_mode()

model.compile(loss=tf.losses.CategoricalCrossentropy(from_logits = True),
              optimizer='adam',
              metrics=['acc'])

class_weight = {0: 2., 1: 1., 2: 1.}

#%%
# checkpoint_filepath = 'drive/MyDrive/saved_models/' + name_arch + '_model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= os.path.join(models_path, base_model_name + '-num_layers_' + str(num_layers) + '-epoch_{epoch:02d}-val_acc_{val_acc:.3f}.h5'),
    monitor='val_acc',
    mode='max',
    save_best_only=True)

callback_stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#%%
# Train the model
history = model.fit(
                    x=traingen, 
                    batch_size=batch_size, 
                    epochs=20, 
                    verbose='auto',
                    callbacks=[model_checkpoint_callback, callback_stop_early],
                    validation_data=valgen, 
                    shuffle=True,
                    class_weight=class_weight
                    )


loss, acc = model.evaluate(x=valgen) # Evaluate to get loss and accuracy of validation

#%%
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(plots_path, f'acc_plot_{base_model_name}-num_layers_{num_layers}-val_acc_{acc:.3f}.png'))
plt.show()

# Loss Curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation Loss'], loc='best')
plt.savefig(os.path.join(plots_path, f'loss_plot_{base_model_name}-num_layers_{num_layers}-val_acc_{acc:.3f}.png'))
plt.show()

#%% Confusion matrix
n_batches = len(valgen)

cm = confusion_matrix(
    np.concatenate([np.argmax(valgen[i][1], axis=1) for i in range(n_batches)]),    
    np.argmax(model.predict(valgen, steps=n_batches), axis=1) 
)

#%% Math of all the metrics remaining

precisions = []
recalls = []
specificities = []
tps = []
fps = []
fns = []
tns = []

for x in range(0,3):
    tp = cm[x,x]
    fp = np.sum(cm[:,x]) - cm[x,x]
    fn = np.sum(cm[x,:]) - cm[x,x]
    tn = np.sum(cm) - (tp + fp + fn)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)

    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
    tns.append(tn)
    precisions.append(precision)
    recalls.append(recall)
    specificities.append(specificity)


#%% To save all the data as CVS

data = [base_model_name, num_layers, acc, loss, tns[0], fps[0], fns[0], tps[0], recalls[0], precisions[0], specificities[0],
        tns[1], fps[1], fns[1], tps[1], recalls[1], precisions[1], specificities[1],
        tns[2], fps[2], fns[2], tps[2], recalls[2], precisions[2], specificities[2],]

df = pd.DataFrame([data], columns=['base_model_name','num_layers','val_acc','val_loss',
                                'tn0','fp0','fn0','tp0','recall0','precision0','specificity0',
                                'tn1','fp1','fn1','tp1','recall1','precision1','specificity1',
                                'tn2','fp2','fn2','tp2','recall2','precision2','specificity2'])

df.to_csv(os.path.join(csvs_path,'multi_experiments.csv'), mode='a', index=False, header=False)

hist_df = pd.DataFrame(history.history)
hist_csv_file = os.path.join(csvs_path, f'history_{base_model_name}-num_layers_{num_layers}-val_acc_{acc}-val_loss_{loss}-experiment.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)



















