import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import random
import labelling
import circledetection
import outliers
import linerdefects_gradient

def test_inner_circle():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        blobs = labelling.bestLabellingTestGradient(img)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))
                        #cv2.circle(img, (int(y), int(x)), int(r), (0, 0, 255), 1)
                        #cv2.circle(img, (int(y), int(x)), 2, (0, 255, 0), 1)

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
            cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def test_outer_circle():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        blobs = labelling.bestLabellingTestGradient(img)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r > 200 and x > 0 and y > 0:
                        circles.append((x, y, r, n))
                        #cv2.circle(img, (int(y), int(x)), int(r), (0, 0, 255), 1)
                        #cv2.circle(img, (int(y), int(x)), 2, (0, 255, 0), 1)

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
            cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def test_edge():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        ret, thresh1 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

        dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
        #dx = np.uint8(np.absolute(dx))

        dest = np.zeros(img.shape, np.float32)
        dest = cv2.normalize(dx, dest, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U)

        cv2.imshow("image", img)
        cv2.imshow("dx", dest)


        #cv2.imshow("image", img)
        #histr = cv2.calcHist([img], [0], None, [256], [0,256])
        #plt.plot(histr)
        #plt.show()

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #magnitude = linerdefects_gradient.calcMagnitude(img)
        #optimalThreshold, magnitude = cv2.threshold(magnitude, 1, 255, cv2.THRESH_OTSU)

        #cv2.imshow("magnitude", magnitude)
        #histr = cv2.calcHist([magnitude], [0], None, [256], [0,256])
        #plt.plot(histr)
        #plt.show()
        #print("OptimalThreshold: " + str(optimalThreshold))

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=1)
        ##optimalThreshold, laplacian = cv2.threshold(laplacian, 1, 255, cv2.THRESH_OTSU)
        ##cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY, laplacian)

        ##print("OptimalThreshold: " + str(optimalThreshold))
        #cv2.imshow("laplacian", laplacian)
        #histr = cv2.calcHist([laplacian], [0], None, [256], [0,256])
        #plt.plot(histr)
        #plt.show()

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #edges = cv2.Canny(img, 40, 100, apertureSize=3, L2gradient=True)

        #cv2.imshow("edges", edges)
        #histr = cv2.calcHist([edges], [0], None, [256], [0,256])
        #plt.plot(histr)
        #plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_outer_with_binarization():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        #hist = cv2.calcHist([img], [0], None, [256], [0,256])
        #plt.plot(hist)
        #plt.show()

        ret, thresh1 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        #thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #ret2, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #print ('threshold found: ' + str(ret2))
                
        cv2.imshow('caps/' + file + ' circles standard', thresh1)
        #cv2.imshow('caps/' + file + ' circles adaptive', thresh2)
        #cv2.imshow('caps/' + file + ' circles otsu', thresh3)

        edge_outer_circle = cv2.Canny(thresh1, 127, 254)

        cv2.imshow('caps/' + file + ' circles edges', edge_outer_circle)

        #circles = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, 100, param1=254, param2=20, minRadius=0, maxRadius=0)

        blobs = labelling.bestLabellingTestGradient(thresh1)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    #if r > 200 and x > 0 and y > 0:
                        circles.append((x, y, r, n))
                        #cv2.circle(img, (int(y), int(x)), int(r), (0, 0, 255), 1)
                        #cv2.circle(img, (int(y), int(x)), 2, (0, 255, 0), 1)

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
            cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #circles = np.uint16(np.around(circles))
        
        #for circle in circles[0,:]:
            # draw the outer circle
            #cv2.circle(img, (round(circle[0]), round(circle[1])), round(circle[2]), (0,255,0), 2)
            # draw the center of the circle
            #cv2.circle(img, (round(circle[0]), round(circle[1])), 2, (0,0,255), 3)

        #cv2.imshow('caps/' + file + ' circles', img)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def test_missing_liner():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('caps/' + file + ' circles', img)

        ret, thresh1 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

        blobs = labelling.bestLabellingTestGradient(thresh1)
        circles = []
        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r > 200 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (x, y), r)
        else:
            mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], center=None, radius=None)

        avg = np.mean(img[mask])
        print("caps/" + file + " pixel's average: " + str(avg))
        if avg >= 60:
            print("caps/" + file + " has no liner!!")
        else:
            print("caps/" + file + " has liner")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #test_inner_circle()
    #test_outer_circle()
    #test_edge()
    test_outer_with_binarization()
    test_missing_liner()