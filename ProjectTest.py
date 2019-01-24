
import cv2
import numpy as np
import os

# hough trasform
dp = 1
minDist = 20

# WARNING: last 4 parameters must be inserted below by hand
p1 = 0
p2 = 0
rMin = 0
rMax = 0

# ------------------------------------------------------------

# image enhancing
gamma = True
r = 0.6

equalized = False

# ------------------------------------------------------------

# noise reduction (no box filter, it's shit
median = False
mKernel = 5

gaussian = False
gKernel = (5, 5)
sigmaX = 0
sigmaY = 0

bilateral = False
diameter = -1
sigmaColor = 1
sigmaSpace = 0

# they say it's for sharpness enhancement and noise removal
# WARNING: not working because it disappeared
adaptiveBilateral = True
abKernel = 5
aSigmaSpace = 1
aMaxSigmaColor = 0

# ------------------------------------------------------------

for file in os.listdir('./caps'):
    img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
    
    if gamma:
        table = np.array([255 ** (1 - r) * i ** r for i in np.arange(0, 256)]).astype('uint8')
        img = cv2.LUT(img, table)
    elif equalized:
        img = cv2.equalizeHist(img)

    if median:
        img = cv2.medianBlur(img, mKernel)
    elif gaussian:
        img = cv2.GaussianBlur(img, gKernel, sigmaX=sigmaX, sigmaY=sigmaY)
    elif bilateral:
        img = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)
    elif adaptiveBilateral:
        img = cv2.adaptiveBilateralFilter(img, abKernel, aSigmaSpace, maxSigmaColor=aMaxSigmaColor)

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey()

cv2.destroyAllWindows()
