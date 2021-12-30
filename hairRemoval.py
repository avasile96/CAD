import cv2
import skimage as io
import numpy as np

def hairRemoval(img, strength):
    thresh, otsu = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter =(3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filter)
    
    black_hat = cv2.morphologyEx(otsu, cv2.MORPH_BLACKHAT,kernel)
    inpaint_img = cv2.inpaint(img, black_hat, 7, flags=cv2.INPAINT_TELEA)
    if strength:
        for i in range(1,strength):
            black_hat = cv2.morphologyEx(inpaint_img, cv2.MORPH_BLACKHAT,kernel)
            inpaint_img = cv2.inpaint(inpaint_img, black_hat, 7, flags=cv2.INPAINT_TELEA)
        
    
    return inpaint_img

if __name__ == '__main__':   
    img_path = 'G:\\DISCO_DURO\\Mixto\\Subjects\\GitHub\\dataset2\\train\\bkl\\bkl0092.jpg'
    img = cv2.imread(img_path)
    cv2.imshow('original', img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = hairRemoval(img_gray, 2)
    cv2.imshow('processed', img2)
    cv2.waitKey()
    
    
    # thresh, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # filter =(11, 11)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filter)
    # black_hat = cv2.morphologyEx(otsu, cv2.MORPH_BLACKHAT,kernel)
    # inpaint_img = cv2.inpaint(img, black_hat, 7, flags=cv2.INPAINT_TELEA)