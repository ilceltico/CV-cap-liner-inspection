import numpy as np
import cv2
import math
import os

def binarize(img):
    #hist = cv2.calcHist([img], [0], None, [256], [0,256])
    #plt.plot(hist)
    #plt.show()

    #ret, thresh_binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)    #using this one instead of otsu we can skip closing, but we are less robust

    ret, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print ('threshold found: ' + str(ret)) #print the threshold found with otsu (ret)
                
    #cv2.imshow('cap binary', thresh_binary)
    #cv2.imshow('cap binary (otsu)', thresh_otsu)
        
    kernel = np.ones((7,7), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))    #using an elliptical kernel shape (obtain the same result)
    closing = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('closing', closing)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return closing

if __name__ == '__main__':
    img = cv2.imread('caps/d_19.bmp', cv2.IMREAD_GRAYSCALE)
    binarize(img)