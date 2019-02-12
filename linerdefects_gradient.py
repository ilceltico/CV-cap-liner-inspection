import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import random
import labelling
import circledetection
import outliers
#from sympy import Point

def calcMagnitude(img):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    #print(dx)
    #print(dy)
    res = cv2.magnitude(dx, dy)
    ##cv2.imshow("res", res)
    #res = cv2.normalize(res, 0.0, 255.0, cv2.CV_8U)
    #print(res)
    dx = np.uint8(np.absolute(dx))
    dy = np.uint8(np.absolute(dy))
    res = np.uint8(np.absolute(res))
    #cv2.imshow("dx", dx)
    #cv2.imshow("dy", dy)
    #cv2.imshow("res", res)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return res

#@profile
#def checkCollinear(x, y):
#    points = zip(x, y)
#    res = Point.is_collinear(*points)
#    return res

#@profile
def pixelAverage():
    for file in os.listdir('./caps'):
        #t1 = cv2.getTickCount()

        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        magnitude = calcMagnitude(img)
        sum = cv2.sumElems(magnitude)
        num = cv2.countNonZero(magnitude)
        average = np.mean(magnitude)
        average2 = sum[0]/num
        #t2 = cv2.getTickCount()
        #time = (t2 - t1)/ cv2.getTickFrequency()

        cv2.imshow("caps/" + file, magnitude)
        print("caps/" + file + " pixel's average: " + str(average))
        print("caps/" + file + " pixel's non zero sum: " + str(sum))
        print("caps/" + file + " pixel's non zero count: " + str(num))
        print("caps/" + file + " pixel's average2: " + str(average2))
        #time = (t2 - t1)/ cv2.getTickFrequency()
        #print ('time: ' + str(time))
        
        print ('----------------------------------------')
        cv2.waitKey()
        cv2.destroyAllWindows()

#@profile
def pixelAverageMask(mask):
    for file in os.listdir('./caps'):
        #t1 = cv2.getTickCount()

        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        magnitude = calcMagnitude(img)
        sum = cv2.sumElems(magnitude[mask])
        num = cv2.countNonZero(magnitude[mask])
        average = np.mean(magnitude[mask])
        average2 = sum[0]/num
        #t2 = cv2.getTickCount()
        #time = (t2 - t1)/ cv2.getTickFrequency()

        cv2.imshow("caps/" + file, magnitude)
        print("caps/" + file + " pixel's average: " + str(average))
        print("caps/" + file + " pixel's non zero sum: " + str(sum))
        print("caps/" + file + " pixel's non zero count: " + str(num))
        print("caps/" + file + " pixel's average2: " + str(average2))
        #time = (t2 - t1)/ cv2.getTickFrequency()
        #print ('time: ' + str(time))
        
        print ('----------------------------------------')
        cv2.waitKey()
        cv2.destroyAllWindows()

def circularmask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

#@profile
def test():
    goodCap = cv2.imread("./caps/g_04.bmp", cv2.IMREAD_GRAYSCALE)
    incompleteLiner = cv2.imread("./caps/d_18.bmp", cv2.IMREAD_GRAYSCALE)
    missingLiner = cv2.imread("./caps/d_31.bmp", cv2.IMREAD_GRAYSCALE)
    goodCapMagnitude = calcMagnitude(goodCap)
    incompleteLinerMagnitude = calcMagnitude(incompleteLiner)
    missingLinerMagnitude = calcMagnitude(missingLiner)

    cv2.imshow("Good Cap Magnitude", goodCapMagnitude)
    cv2.imshow("Incomplete Liner Magnitude", incompleteLinerMagnitude)
    cv2.imshow("Missing Liner Magnitude", missingLinerMagnitude)

    print("Good Cap Magnitude average of pixels: " + str(np.mean(goodCapMagnitude)))
    print("Incomplete Liner Magnitude average of pixels: " + str(np.mean(incompleteLinerMagnitude)))
    print("Missing Liner Magnitude average of pixels: " + str(np.mean(missingLinerMagnitude)))

    temp = cv2.fastNlMeansDenoising(goodCapMagnitude, None, 15, 7, 21)
    cv2.imshow("temp", temp)

    cv2.waitKey()
    cv2.destroyAllWindows()

    blobs = labelling.bestLabellingTestGradient(temp)

    #for i in blobs:
    #    print (i)

    #for i in range(0, len(blobs)):
    #    print (i)
    #    print (blobs[i])

    #print (len(blobs))

    img = cv2.imread('./caps/g_04.bmp', cv2.IMREAD_COLOR)
    cv2.imshow('original', img)
    cv2.waitKey()

    circles = []

    for blob in blobs:
        #if len(blob[0]) > 1 and checkCollinear(blob[0], blob[1]) == False:
        #if len(blob[0]) > 1:
        if len(blob[0]) > 2:
            x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
            if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                circles.append((x, y, r, n))

    #print (len(circles))
    x, y, r = outliers.outliersElimination(circles, (100, 100))
    cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
    cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
    cv2.imshow('circles', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #for i in range(0, 10):
        #test()

    #for i in range(0, 20):
    #    mask = circularmask(576, 768, center=None, radius=None)
    #    pixelAverageMask(mask)

    #    pixelAverage()

    pixelAverage()
    mask = circularmask(576, 768, center=None, radius=None)
    pixelAverageMask(mask)