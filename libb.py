# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:52:54 2021

@author: vasil
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import gc
import cv2


# Harris Corners
dst = cv2.cornerHarris(x_val_arr[100,:,:,2], blockSize=2, ksize=3, k=0.04)

# dilate to mark the corners
dst = cv2.dilate(dst, None)

# Multiply the image with a gaussian only to get central points
x_val_arr[100,:,:,2][dst > 0.01 * dst.max()] = 255

cv2.imshow('haris_corner', x_val_arr[100,:,:,2])
cv2.waitKey()
