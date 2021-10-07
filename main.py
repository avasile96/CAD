# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:34:02 2021

@author: usuari
"""
import os


source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 



input_img_paths = []
for label in os.listdir(os.path.join(dataset_dir, 'train')):
    if os.path.isdir(os.path.join(dataset_dir, 'train', label)):
        for fname in os.listdir(os.path.join(dataset_dir, 'images', patient_index)):
            if fname.endswith(".jpg") and not fname.startswith("."):
                input_img_paths.append(os.path.join(dataset_dir, 'images', patient_index, fname))

target_img_paths = [
        os.path.join(dataset_dir, 'groundtruth', fname)
        for fname in os.listdir(os.path.join(dataset_dir, 'groundtruth'))
        if fname.endswith(".tiff") and not fname.startswith(".")]