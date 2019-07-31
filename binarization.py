import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt

def binarize(img):
    #hist = cv2.calcHist([img], [0], None, [256], [0,256])
    #plt.plot(hist)
    #plt.show()

    #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_TOZERO)

    ret, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print ('threshold found: ' + str(ret)) #print the threshold found with otsu (ret)
                
    #cv2.imshow('cap binary', thresh_binary)
    #cv2.imshow('cap binary (otsu)', thresh_otsu)
        
    kernel = np.ones((7,7), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))    #using an elliptical kernel shape (obtain the same result)
    closing = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('closing', closing)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return closing

def is_circle(img):
    binary = binarize(img)
    #cv2.imshow('binary', binary)

    # Contours are simply a curve joining all the points (along the boundary), having same color or intensity.
    #first argument: image
    #second argument: contour retrieval mode.
    #third argument: contour approximation method. Two possibilities: 
    #   cv2.CHAIN_APPROX_NONE: all the boundary points are stored.
    #   cv2.CHAIN_APPROX_SIMPLE: removes all redundant points and compresses the contour, saving memory. E.g. for a line store only the two end points.
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    #M = cv2.moments(cnt)
    #print(M)

    #   area
    #area_contour = M['m00']
    area_non_zero = len(np.nonzero(binary)[0])

    #print('area_contour: ' + str(area_contour))
    #print('area_non_zero: ' + str(area_non_zero))

    #   perimeter
    #second argument specify whether shape is a closed contour (if passed True), or just a curve
    perimeter = cv2.arcLength(cnt, True)
    #perimeter = cv2.arcLength(cnt, False)

    #print('perimeter: ' + str(perimeter))

    #   baricenter (centroids). Can be computed as m10 / m00 and m01 / m00, knowing that m10 and m01 are sum(i) and sum(j), and m00 is the area.
    #cx = M['m10']/M['m00']
    #cy = M['m01']/M['m00']

    #print('cx ' + str(cx))
    #print('cy ' + str(cy))

    #   compactness (i.e. form factor). For a circle, it must be close to 1 to be a circle (normalized with 4pi)
    #print('area_contour - perimeter: ' + str(np.round(perimeter**2 / area_contour)/(4*math.pi)))
    #print('area_non_zero - perimeter: ' + str((perimeter**2 / area_non_zero)/(4*math.pi)))
    
    if np.round((perimeter**2 / area_non_zero)/(4*math.pi)).astype("int") == 1 :
        #print('the cap is a circle')
        return True
    else:
        return False

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('caps/d_19.bmp', cv2.IMREAD_GRAYSCALE)
    #binarize(img)
    is_circle(img)