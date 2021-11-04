# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:52:54 2021

@author: vasil
"""

import numpy as np
import cv2
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.filters import threshold_triangle
from skimage.morphology import area_opening
from sklearn.cluster import KMeans
import skimage as io
from hairRemoval import hairRemoval

# # Harris Corners
# dst = cv2.cornerHarris(x_val_arr[100,:,:,2], blockSize=2, ksize=3, k=0.04)

# # dilate to mark the corners
# dst = cv2.dilate(dst, None)

# # Multiply the image with a gaussian only to get central points
# x_val_arr[100,:,:,2][dst > 0.01 * dst.max()] = 255

# cv2.imshow('haris_corner', x_val_arr[100,:,:,2])
# cv2.waitKey()


# def makeGaussian(size = 450, fwhm = 100, center=[225,225]):
#     """ Make a square gaussian kernel.

#     size is the length of a side of the square
#     fwhm is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """

#     x = np.arange(0, size, 1, int)
#     y = x[:,np.newaxis]

#     if center is None:
#         x0 = y0 = size // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]
    
#     gauss = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
#     # print(np.amax(gauss))
    
#     # Applying padding
#     mask = np.zeros([450,600])
#     mask[:,74:524] = gauss
    
#     # cv2.imshow('msk', mask)
    
#     return mask


def vignette(input_image):
    # Extracting the height and width of an image 
    height, width = input_image.shape[:2] 
   
    X_resultant_kernel = cv2.getGaussianKernel(width,300) 
    Y_resultant_kernel = cv2.getGaussianKernel(height,225) 
        
    #generating resultant_kernel matrix 
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        
    #creating mask and normalising by using np.linalg
    # function
    mask = resultant_kernel / np.linalg.norm(resultant_kernel) 
    output = np.copy(input_image)
        
    # applying the mask to each channel in the input image
    output = output * 1/mask # positive vignette because of the 1/
            
    #displaying the orignal image   
    # cv2.imshow('Original', input_image)
        
    #displaying the vignette filter image 
    # cv2.imshow('VIGNETTE', output)
    return output

def preprocessing(input_image):


    input_image = input_image

    tophat_img = hairRemoval(input_image, strength=2)

    # cv2.imshow("original", input_image)
    # cv2.imshow("tophat", tophat_img)
    # cv2.waitKey(5000)

    # cv2.imshow("tophat", tophat_img)

    # Filtering out the Noise
    img_filtered = anisotropic_diffusion(tophat_img)
    img_filtered = np.array((img_filtered/np.max(img_filtered))*255,dtype=np.uint8)
    # cv2.imshow("tophat_filtered", img_filtered)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    equalized = clahe.apply(img_filtered)
    # cv2.imshow("eq", equalized)

    # Mulitplying with gaussian for focus
    center_image = vignette(equalized)
    center_image = np.array((center_image/np.max(center_image))*255,dtype=np.uint8)
    # cv2.imshow("center_image", center_image)
    # io.imsave('sal.png', center_image)

    # Thresholding
    th_logical = center_image < threshold_triangle(center_image, nbins = 2)
    # seg_im = th_logical * center_image
    # cv2.imshow("seg_im", seg_im)

    # Binary Morphology
    op_img = area_opening(np.uint8(th_logical))*255 # opening
    # cv2.imshow("op_img", op_img)

    kernel = np.ones((5,5),np.uint8)
    dil_img = cv2.dilate(op_img,kernel,iterations = 1) # closing
    # cv2.imshow("dil_img", dil_img)

    # Deleting border regions
    mask1 = np.zeros_like(dil_img)
    mask1 = cv2.circle(mask1, (300,225), 300, (255,255,255), -1)
    # cv2.imshow("mask1", mask1)

    seg_mask = dil_img*mask1/255
    seg_mask = np.uint8(seg_mask>0)
    # cv2.imshow("seg_mask", seg_mask*255)

    # Getting the processed image
    segim = seg_mask*center_image
    # cv2.imshow("segim", segim)
    return segim
    
if __name__ == '__main__':
    pass
    
    
    
    