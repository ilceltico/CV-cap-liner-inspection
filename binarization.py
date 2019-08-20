import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt

def binarize(img):
    # hist = cv2.calcHist([img], [0], None, [256], [0,256])
    # plt.plot(hist)
    # plt.show()

    # ret, thresh_binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_TOZERO)

    #Otsu's Thresholding
    ret, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print ('threshold found: ' + str(ret)) #print the threshold found with otsu (ret)
                
    #cv2.imshow('cap binary', thresh_binary)
    #cv2.imshow('cap binary (otsu)', thresh_otsu)
        
    #Morphological Closing operation
    kernel = np.ones((7,7), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))    #using an elliptical kernel shape (obtain the same result)
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
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #Should find 4-connected perimeters

    cnt = contours[0]
    # print(cnt)
    # print(cnt[0])
    # print(cnt[0][0])
    # M = cv2.moments(cnt)
    # print(M)
    # vis = np.zeros((len(img),len(img[0])), np.uint8)
    # cv2.drawContours( vis, contours, (-1, 2)[3 <= 0], (128,255,255),
    #         1, cv2.LINE_AA, hierarchy, abs(3) )
    # cv2.imshow('contours', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #   area
    #area_contour = M['m00']
    #area_non_zero = len(np.nonzero(binary)[0])

    #print('area_contour: ' + str(area_contour))
    #print('area_non_zero: ' + str(area_non_zero))

    #   perimeter
    #second argument specify whether shape is a closed contour (if passed True), or just a curve
    perimeter = cv2.arcLength(cnt, True)
    #perimeter = cv2.arcLength(cnt, False)

    #Haralick's Circularity

    moments = cv2.moments(cnt)
    # print(cv2.moments(img)["m00"])
    # print(cv2.moments(binary)["m00"])
    # print(len(binary))
    # print(len(binary[0]))
    # print(cv2.moments(binary)["m10"])
    i_b = int(moments["m10"] / moments["m00"])
    j_b = int(moments["m01"] / moments["m00"])
    # print([i_b, j_b])
    diff = cnt-np.full_like(cnt,[i_b, j_b])
    # print(ar)
    diff = (diff).reshape(len(diff),2)
    # print(ar)
    # print(np.linalg.norm(ar, axis=1))
    # print(len(ar[0]))
    distances_from_bary = np.linalg.norm(diff, axis=1)
    average_distance = np.sum(distances_from_bary) / len(distances_from_bary);
    # print(average_distance)
    # print(area_non_zero)
    # print(moments["m00"])

    variance = np.sum(np.square(distances_from_bary - average_distance)) / len(distances_from_bary);
    # print(std_dev)

    haralick_circularity = average_distance / variance
    print('Circularity: ' + str(haralick_circularity))


    # erosion
    #kernel = np.ones((3,3),np.uint8)
    #erosion = cv2.erode(binary, kernel, iterations=1)
    #contour = binary - erosion
    #cv2.imshow('contour', contour)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #perimeter = len(np.nonzero(contour)[0])

    # dilation
    #kernel = np.ones((3,3),np.uint8)
    #dilation = cv2.dilate(binary, kernel, iterations=1)
    #contour = dilation - binary
    #cv2.imshow('contour', contour)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #perimeter = len(np.nonzero(contour)[0])

    print('perimeter: ' + str(perimeter))

    #   baricenter (centroids). Can be computed as m10 / m00 and m01 / m00, knowing that m10 and m01 are sum(i) and sum(j), and m00 is the area.
    #cx = M['m10']/M['m00']
    #cy = M['m01']/M['m00']

    #print('cx ' + str(cx))
    #print('cy ' + str(cy))

    #   compactness (i.e. form factor). For a circle, it must be close to 1 to be a circle (normalized with 4pi)
    #print('area_contour - perimeter: ' + str(np.round(perimeter**2 / area_contour)/(4*math.pi)))
    #print('area_non_zero - perimeter: ' + str((perimeter**2 / area_non_zero)/(4*math.pi)))
    
    # if np.round((perimeter**2 / area_non_zero)/(4*math.pi)).astype("int") == 1 :
    #     #print('the cap is a circle')
    #     return True
    # else:
    #     return False

    # Normal circularity
    # tolerance = 0.12
    # if np.abs((perimeter**2 / area_non_zero)/(4*math.pi) - 1) <= tolerance :
    #     #print('the cap is a circle')
    #     return True
    # else:
    #     return False

    # Test Haralick's circularity
    minimum = 200
    if haralick_circularity >= minimum :
        #print('the cap is a circle')
        return True
    else:
        return False

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        binarize(img)