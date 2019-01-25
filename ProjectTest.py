import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

NOEQ = 0
LINEAR = 1
GAMMA = 2
EQUALIZATION = 3

NOFILTER = 0
MEDIAN = 1
GAUSSIAN = 2
BILATERAL = 3
NONLOCALMEANS = 4

# hough trasform
dp = 1
minDist = 3
param1 = 100
param2 = 100
minRadius = 0
maxRadius = 0

## ------------------------------------------------------------

## showing histogram
#showHistogram = False

## ------------------------------------------------------------

## image enhancing
#linear = True
#lowerPercentile = 0
#upperPercentile = 0

#gamma = False
#r = 0.5

#equalized = False

## ------------------------------------------------------------

## noise reduction (no box filter, it's shit
#median = False
#mKernel = 5

#gaussian = False
#gKernel = (5, 5)
#sigmaX = 0
#sigmaY = 0

#bilateral = False
#diameter = -1
#sigmaColor = 5
#sigmaSpace = 5

#nonLocalMean = False
#h = 10
#templateWindowSize = 7
#searchWindowSize = 21

## they say it's for sharpness enhancement and noise removal
## WARNING: not working because it disappeared
#adaptiveBilateral = False
#abKernel = 5
#aSigmaSpace = 1
#aMaxSigmaColor = 0

## ------------------------------------------------------------

#for file in os.listdir('./caps'):
#    img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
#    if linear:
#        hist = cv2.calcHist([img], [0], None, [256], [0,256])
#        rows, cols = img.shape 
#        total = rows * cols
#        temp = 0
#        lower = 0

#        while temp * 100 <= lowerPercentile * total:
#            temp += hist[lower]
#            lower += 1

#        temp = 0
#        upper = 255

#        while temp * 100 <= upperPercentile * total:
#            temp += hist[upper]
#            upper -= 1

#        table = [None] * 256

#        for i in range(0, 256):
#            new = 255 * (i - lower) / (upper - lower)
#            if new > 255:
#                new = 255
#            if new < 0:
#                new = 0
#            table[i] = new

#        for row in range(0, rows):
#            for col in range(0, cols):
#                px = img.item(row, col)
#                img.itemset((row, col), table[px])
#    elif gamma:
#        table = np.array([255 ** (1 - r) * i ** r for i in np.arange(0, 256)]).astype('uint8')
#        img = cv2.LUT(img, table)
#    elif equalized:
#        img = cv2.equalizeHist(img)

#    if median:
#        img = cv2.medianBlur(img, mKernel)
#    elif gaussian:
#        img = cv2.GaussianBlur(img, gKernel, sigmaX=sigmaX, sigmaY=sigmaY)
#    elif bilateral:
#        img = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)
#    elif adaptiveBilateral:
#        img = cv2.adaptiveBilateralFilter(img, abKernel, aSigmaSpace, maxSigmaColor=aMaxSigmaColor)
#    elif nonLocalMean:
#        img = cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)  

#    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

#    if circles is not None:
#        circles = np.uint16(np.around(circles))
#        for i in circles[0,:]:
#            # draw the outer circle
#            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
#            # draw the center of the circle
#            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

#    cv2.imshow('detected circles', cimg)
    
#    if showHistogram:
#        histr = cv2.calcHist([img], [0], None, [256], [0,256])
#        plt.plot(histr)
#        plt.show()
#        print(histr)
    
#    cv2.waitKey()

#    #temporary
#    break

#cv2.destroyAllWindows()

def houghPlot(img, showHistogram, dp, minDist, param1, param2, minRadius, maxRadius, histEqType = NOEQ, linearPercentiles = [0,0], gammaR = 1, filterType = NOFILTER, medianKernelDim = 5, gaussianKernelTuple = (5,5), gaussianSigmaX = 0, gaussianSigmaY = 0, bilateralDiameter = -1, bilateralSigmaColor = 5, bilateralSigmaSpace = 5, nonLocalMeansH = 10, nonLocalMeansKernelDim = 7, nonLocalMeansSearchWindowDim = 21):
    if histEqType == LINEAR:
        hist = cv2.calcHist([img], [0], None, [256], [0,256])
        rows, cols = img.shape 
        total = rows * cols
        temp = 0
        lower = 0

        while temp * 100 <= linearPercentiles[0] * total:
            temp += hist[lower]
            lower += 1

        temp = 0
        upper = 255

        while temp * 100 <= linearPercentiles[1] * total:
            temp += hist[upper]
            upper -= 1

        table = [None] * 256

        for i in range(0, 256):
            new = 255 * (i - lower) / (upper - lower)
            if new > 255:
                new = 255
            if new < 0:
                new = 0
            table[i] = new

        for row in range(0, rows):
            for col in range(0, cols):
                px = img.item(row, col)
                img.itemset((row, col), table[px])
    elif histEqType == GAMMA:
        table = np.array([255 ** (1 - gammaR) * i ** gammaR for i in np.arange(0, 256)]).astype('uint8')
        img = cv2.LUT(img, table)
    elif histEqType == EQUALIZATION:
        img = cv2.equalizeHist(img)

    if filterType == MEDIAN:
        img = cv2.medianBlur(img, medianKernelDim)
    elif filterType == GAUSSIAN:
        img = cv2.GaussianBlur(img, gaussianKernelTuple, sigmaX=gaussianSigmaX, sigmaY=gaussianSigmaY)
    elif filterType == BILATERAL:
        img = cv2.bilateralFilter(img, bilateralDiameter, bilateralSigmaColor, bilateralSigmaSpace)
    elif filterType == NONLOCALMEANS:
        img = cv2.fastNlMeansDenoising(img, None, nonLocalMeansH, nonLocalMeansKernelDim, nonLocalMeansSearchWindowDim)  

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    plt.figure()
    if showHistogram:
        histr = cv2.calcHist([img], [0], None, [256], [0,256])
        plt.subplot(121), plt.plot(histr)
        plt.subplot(122), plt.imshow(cimg)
    else:
        plt.imshow(cimg)
    



houghPlot(cv2.imread("caps/d_16.bmp", cv2.IMREAD_GRAYSCALE), True, 1, 3, 100, 100, 0, 0, histEqType = LINEAR)
houghPlot(cv2.imread("caps/d_16.bmp", cv2.IMREAD_GRAYSCALE), True, 1, 3, 100, 100, 0, 0)
plt.show()