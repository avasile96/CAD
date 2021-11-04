# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:52:54 2021

@author: vasil
"""

import numpy as np
import cv2


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
    
if __name__ == '__main__':
    pass
    
    
    
    