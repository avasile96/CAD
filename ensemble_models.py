"""
Created on Mon Dec 13 12:33:42 2021

@author: 52331
"""
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

type_class = 'multi' # 'binary' or 'multi'

if type_class == 'multi':
    dataset_dir = os.path.join(project_dir, 'dataset2') 
    classification_type = 'multi' 
    classes_num = 3
    THRESHOLD = 740
    
if type_class == 'binary':
    dataset_dir = os.path.join(project_dir, 'dataset') 
    classification_type = 'binary'
    classes_num = 2
    THRESHOLD = 845

# Directores to save the results for binary and multiclass
DL_folder = os.path.join(source_dir, 'DL_folders')
csvs_path = os.path.join(DL_folder, classification_type, 'CSVs')
models_path = os.path.join(DL_folder, classification_type, 'models')
plots_path = os.path.join(DL_folder, classification_type, 'plots')


# Generator Parameters
img_size = (450, 600, 3) # RGB imges!
batch_size = 2

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

    def __init__(self, batch_size, img_size, input_img_paths, img_label, classification_type):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        random.shuffle(self.input_img_paths)
        self.img_label = img_label
        self.shuffle = True
        self.classification_type = classification_type

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
            
            if classification_type == 'binary':
                if path[-10:-8] == 'ls':
                    y[j] = 1
                else:
                    y[j] = 0
                    
            if classification_type == 'multi':
                if path[-11:-8] == 'bcc':
                  y[j] = 0
                elif path[-11:-8] == 'bkl':
                  y[j] = 1
                else:
                  y[j] = 2   
                
        return x, tf.keras.utils.to_categorical(y, num_classes=classes_num)

traingen = SkinImageDatabase(batch_size, img_size, train_img_paths, img_label, classification_type)
valgen = SkinImageDatabase(batch_size, img_size, val_img_paths, val_label, classification_type)

gc.collect()

#%% Acquicision of models with high accuracy 

if classification_type == 'binary':
    
    models = []
    for model in os.listdir(models_path):
        if int(model[-6:-3]) >= THRESHOLD: # Resnet50 acc:853, Resnet50 acc=873, vgg16 acc=846
            models.append(tf.keras.models.load_model(os.path.join(models_path, model)))


if classification_type == 'multi':
    
    models = []
    for model in os.listdir(models_path):
        if int(model[-6:-3]) >= THRESHOLD:
            models.append(tf.keras.models.load_model(os.path.join(models_path, model)))


gc.collect()
# model0 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\basic_DenseNet121_model_val_acc_81.h5') 
# model1 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\dropout_5_VGG16_model_val_acc_84.h5')
# model2 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\dropout_5_DenseNet121_model_val_acc_818.h5') 
# model3 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\basic_VGG16_model_val_acc_82.h5') 
# model4 = tf.keras.models.load_model('C:\\Users\\52331\\Downloads\\dropout_5_VGG16_model_val_acc_83.h5') 
# models = [model0, model1, model2, model3, model4]

#%% Prediction
# make predictions
# with tf.device('/gpu:0'):
yhats = [model.predict(valgen) for model in models]
yhats = np.array(yhats)

# sum across ensembles
summed = np.sum(yhats, axis=0)
# argmax across classes
outcomes = np.argmax(summed, axis=1)


#%% Accuracy

n_batches = len(valgen)

true_labels = np.concatenate([np.argmax(valgen[i][1], axis=1) for i in range(n_batches)])
acc = accuracy_score(true_labels, outcomes)


#%% Confusion matrix and metrics depending on the classification (binary or multiclass)

if classification_type == 'binary':
    
    tn, fp, fn, tp = confusion_matrix(true_labels, outcomes).ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)

    data = ['ensemble_models', len(models), acc, '-', tn, fp, fn, tp, recall, precision, specificity]
    df = pd.DataFrame([data],columns=['base_model_name','num_layers','val_acc',
                                      'val_loss','tn','fp','fn','tp','recall','precision','specificity'])
    
    df.to_csv(os.path.join(csvs_path,'binary_experiments.csv'), mode='a', index=False, header=False)

    
if classification_type == 'multi':
    
    cm = confusion_matrix(true_labels, outcomes)

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
    
    data = ['ensemble_models', len(models), acc, '-', tns[0], fps[0], fns[0], tps[0], recalls[0], precisions[0], specificities[0],
            tns[1], fps[1], fns[1], tps[1], recalls[1], precisions[1], specificities[1],
            tns[2], fps[2], fns[2], tps[2], recalls[2], precisions[2], specificities[2],]
    
    df = pd.DataFrame([data], columns=['base_model_name','num_layers','val_acc','val_loss',
                                    'tn0','fp0','fn0','tp0','recall0','precision0','specificity0',
                                    'tn1','fp1','fn1','tp1','recall1','precision1','specificity1',
                                    'tn2','fp2','fn2','tp2','recall2','precision2','specificity2'])
    
    df.to_csv(os.path.join(csvs_path,'multi_experiments.csv'), mode='a', index=False, header=False)




