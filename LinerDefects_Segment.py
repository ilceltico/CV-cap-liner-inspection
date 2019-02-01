import numpy as np
import cv2
import math
import os

#img1 = cv2.imread('./caps/d_17.bmp', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('./caps/d_19.bmp', cv2.IMREAD_GRAYSCALE)

#cimg1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#cimg2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

#canny1 = cv2.Canny(img1, 50, 100, apertureSize=3, L2gradient=False)
#canny2 = cv2.Canny(img2, 50, 100, apertureSize=3, L2gradient=False)

#cv2.imshow('img1', canny1)
#cv2.imshow('img2', canny2)
#cv2.waitKey()

#segments1 = cv2.HoughLinesP(canny1, 1, math.pi/180, 100)
#segments2 = cv2.HoughLinesP(canny2, 1, math.pi/180, 100)

#if segments1 is not None:
#    for segment in segments1:
#        cv2.line(cimg1, (segment[0][0], segment[0][1]), (segment[0][2], segment[0][3]), (0, 0, 255))

#if segments2 is not None:
#    for segment in segments2:
#        cv2.line(cimg2, (segment[0][0], segment[0][1]), (segment[0][2], segment[0][3]), (0, 0, 255))


#lines1 = cv2.HoughLines(canny1, 1, math.pi/180, 70)
#lines2 = cv2.HoughLines(canny2, 1, math.pi/180, 70)

#rows1, cols1 = img1.shape
#rows2, cols2 = img2.shape

#if lines1 is not None:
#    for line in lines1:
#        rho = line[0][0]
#        theta = line[0][1]
#        if theta < math.pi / 4 or theta > 3 * math.pi / 4:
#            x1 = rho / math.cos(theta)
#            y1 = 0
#            x2 = (rho - rows1 * math.sin(theta)) / math.cos(theta)
#            y2 = rows1
#        else:
#            x1 = 0
#            y1 = rho / math.sin(theta)
#            x2 = cols1
#            y2 = (rho - cols1 * math.cos(theta)) / math.sin(theta)

#        cv2.line(cimg1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))

#if lines2 is not None:
#    for line in lines2:
#        rho = line[0][0]
#        theta = line[0][1]
#        if theta < math.pi / 4 or theta > 3 * math.pi / 4:
#            x1 = rho / math.cos(theta)
#            y1 = 0
#            x2 = (rho - rows2 * math.sin(theta)) / math.cos(theta)
#            y2 = rows2
#        else:
#            x1 = 0
#            y1 = rho / math.sin(theta)
#            x2 = cols2
#            y2 = (rho - cols2 * math.cos(theta)) / math.sin(theta)

#        cv2.line(cimg2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))


#cv2.imshow('seg1', cimg1)
#cv2.imshow('seg2', cimg2)
#cv2.waitKey()

for file in os.listdir('./caps'):
    img = cv2.imread('./caps/' + file, cv2.IMREAD_GRAYSCALE)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canny = cv2.Canny(img, 50, 100, apertureSize=3, L2gradient=False)

    lines = cv2.HoughLines(canny, 1, math.pi/180, 70)

    rows, cols = img.shape
    
    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            if theta < math.pi / 4 or theta > 3 * math.pi / 4:
                x1 = rho / math.cos(theta)
                y1 = 0
                x2 = (rho - rows * math.sin(theta)) / math.cos(theta)
                y2 = rows
            else:
                x1 = 0
                y1 = rho / math.sin(theta)
                x2 = cols
                y2 = (rho - cols * math.cos(theta)) / math.sin(theta)

            cv2.line(cimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))

    cv2.imshow('original', img)
    cv2.imshow('canny', canny)
    cv2.imshow('defects', cimg)
    cv2.waitKey()

cv2.destroyAllWindows()