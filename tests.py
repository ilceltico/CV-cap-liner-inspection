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
import binarization

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

def test_outer_with_binarization():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        binary = binarization.binarize(img)
        #cv2.imshow('caps/' + file + ' binary', binary)

        edges = cv2.Canny(binary, 45, 100, apertureSize=3, L2gradient=True)
        cv2.imshow("edges", edges)
        
        blobs = labelling.bestLabellingGradient(edges)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(img, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r).astype("int"), (0, 255, 0), 1)
            cv2.circle(img, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def test_missing_liner():
    thresh = getThreshold()
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('caps/' + file + ' circles', img)

        binary = binarization.binarize(img)
        #cv2.imshow('caps/' + file + ' binary', binary)

        blobs = labelling.bestLabellingTestGradient(binary)

        circles = []
        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                   circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (x, y), r)
            avg = np.mean(img[mask])
        else:
            #mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], center=None, radius=None)
            avg = np.mean(img)

        #avg = np.mean(img[mask])
        print("caps/" + file + " pixel's average: " + str(avg))
        if avg > thresh:
            print("caps/" + file + " has no liner!!")
        else:
            print("caps/" + file + " has liner")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def getThreshold():
    thresh = 0
    prefixed = [filename for filename in os.listdir('./caps') if filename.startswith("g")]
    for file in prefixed:
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        
        binary = binarization.binarize(img)

        blobs = labelling.bestLabellingTestGradient(binary)

        circles = []
        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                   circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (x, y), r)
        avg = np.mean(img[mask])

        thresh = thresh + avg
    
    return (thresh / len(prefixed)) + 10    #to consider the cap with no liner it must have a big difference with the correct average

def test_inner_liner_magnitude():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('caps/' + file, img)

        magnitude = linerdefects_gradient.calcMagnitude(img)
        cv2.imshow("magnitude", magnitude)
        fast = cv2.fastNlMeansDenoising(magnitude, None, 10, 7, 21)
        cv2.imshow("fast", fast)

        edges = cv2.Canny(fast, 40, 100, apertureSize=3, L2gradient=False)
        cv2.imshow("edges", edges)

        blobs = labelling.bestLabellingTestGradient(fast)

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

def another_inner_circle():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('caps/' + file, img)

        fast = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
        cv2.imshow("fast", fast)

        magnitude = linerdefects_gradient.calcMagnitude(fast)
        cv2.imshow("magnitude", magnitude)

        edges = cv2.Canny(magnitude, 40, 100, apertureSize=3, L2gradient=False)
        cv2.imshow("edges", edges)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        blobs = labelling.bestLabellingTestGradient(magnitude)

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

#@profile
def best_inner_circle():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('caps/' + file, img)

        #histr = cv2.calcHist([img], [0], None, [256], [0,256])
        #plt.plot(histr)
        #plt.show()

        #LINEAR STRETCHING
        #imgOut = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

        #max = img.max()
        #min = img.min()
        #for row in range(0, img.shape[0]):
        #    for col in range(0, img.shape[1]):
        #        pixel = img.item(row, col)
        #        pixelOut = (255 / (max - min))*(pixel - min)
        #        imgOut.itemset((row, col), pixelOut)

        #LINEAR STRETCHING A LOT MORE EFFICIENT
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)

        #cv2.imshow("Stretched image", imgOut)
        #histr = cv2.calcHist([imgOut], [0], None, [256], [0,256])
        #plt.plot(histr)
        #plt.show()

        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        #cv2.imshow("gaussian", gaussian)

        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
        #cv2.imshow("edges", edges)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        blobs = labelling.bestLabellingGradient(edges)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(img, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r).astype("int"), (0, 255, 0), 1)
            cv2.circle(img, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def compare_all_inner_results():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('caps/' + file, img)

        # INNER_CIRCLE
        blobs = labelling.bestLabellingTestGradient(img)

        imgO = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(imgO, (int(y), int(x)), int(r), (0, 255, 0), 1)
            cv2.circle(imgO, (int(y), int(x)), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles', imgO)

        # INNER MAGNITUDE
        magnitude = linerdefects_gradient.calcMagnitude(img)
        fast = cv2.fastNlMeansDenoising(magnitude, None, 10, 7, 21)

        blobs = labelling.bestLabellingTestGradient(fast)

        imgM = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
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
            cv2.circle(imgM, (int(y), int(x)), int(r), (0, 255, 0), 1)
            cv2.circle(imgM, (int(y), int(x)), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles magnitude', imgM)

        # ANOTHER INNER CIRCLE
        fast = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
        magnitude = linerdefects_gradient.calcMagnitude(fast)

        blobs = labelling.bestLabellingTestGradient(magnitude)

        imgA = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(imgA, (int(y), int(x)), int(r), (0, 255, 0), 1)
            cv2.circle(imgA, (int(y), int(x)), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles another', imgA)

        # BEST INNER CIRCLE
        #LINEAR STRETCHING A LOT MORE EFFICIENT THAN CYCLE
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)

        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
        blobs = labelling.bestLabellingGradient(edges)

        imgB = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            cv2.circle(imgB, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r).astype("int"), (0, 255, 0), 1)
            cv2.circle(imgB, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' circles best', imgB)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #test_inner_circle()
    #test_outer_circle()
    #test_outer_with_binarization()
    #test_missing_liner()
    #test_inner_liner_magnitude()
    #another_inner_circle()
    best_inner_circle()
    #compare_all_inner_results()