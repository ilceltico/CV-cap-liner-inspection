import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import random
import labelling
import circledetection
import outliers

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

if __name__ == '__main__':
    goodCap = cv2.imread("./caps/d_16.bmp", cv2.IMREAD_GRAYSCALE)
    incompleteLiner = cv2.imread("./caps/d_18.bmp", cv2.IMREAD_GRAYSCALE)
    missingLiner = cv2.imread("./caps/d_31.bmp", cv2.IMREAD_GRAYSCALE)
    goodCapMagnitude = calcMagnitude(goodCap)
    incompleteLinerMagnitude = calcMagnitude(incompleteLiner)
    missingLinerMagnitude = calcMagnitude(missingLiner)

    cv2.imshow("Good Cap Magnitude", goodCapMagnitude)
    cv2.imshow("Incomplete Liner Magnitude", incompleteLinerMagnitude)
    cv2.imshow("Missing Liner Magnitude", missingLinerMagnitude)

    temp = cv2.fastNlMeansDenoising(goodCapMagnitude, None, 18, 7, 21)
    cv2.imshow("temp", temp)

    cv2.waitKey()
    cv2.destroyAllWindows()

    blobs = labelling.bestLabellingTestGradient(temp)

    #for i in blobs:
    #    print (i)

    #for i in range(0, len(blobs)):
    #    print (i)
    #    print (blobs[i])

    print (len(blobs))

    img = cv2.imread('./caps/d_16.bmp', cv2.IMREAD_COLOR)
    #cv2.imshow('original', img)
    #cv2.waitKey()

    circles = []

    for blob in blobs:
        x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
        #print (x)
        #print (y)
        #print (r)

        if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
            circles.append((x, y, r, n))

    print (len(circles))
    x, y, r = outliers.outliersElimination(circles, (100, 100))
    cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
    cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
    cv2.imshow('circles', img)
    cv2.waitKey()
    cv2.destroyAllWindows()