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
    #dx = np.uint8(np.absolute(dx))
    #dy = np.uint8(np.absolute(dy))
    res = np.uint8(np.absolute(res))
    #cv2.imshow("dx", dx)
    #cv2.imshow("dy", dy)
    #cv2.imshow("res", res)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
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
        summa = cv2.sumElems(magnitude)
        num = cv2.countNonZero(magnitude)
        average = np.mean(magnitude)
        average2 = summa[0]/num
        #t2 = cv2.getTickCount()
        #time = (t2 - t1)/ cv2.getTickFrequency()

        cv2.imshow("caps/" + file, magnitude)
        print("caps/" + file + " pixel's magnitude average: " + str(average))
        print("caps/" + file + " pixel's non zero sum: " + str(summa))
        print("caps/" + file + " pixel's non zero count: " + str(num))
        print("caps/" + file + " pixel's magnitude average2: " + str(average2))
        
        print("caps/" + file + " pixel's average: " + str(np.mean(img)))
        #time = (t2 - t1)/ cv2.getTickFrequency()
        #print ('time: ' + str(time))

        
        #median = cv2.medianBlur(goodCapMagnitude, 5)
        #gaussian = cv2.GaussianBlur(img, gaussianKernelTuple, sigmaX=gaussianSigmaX, sigmaY=gaussianSigmaY)
        #bilateral = cv2.bilateralFilter(img, bilateralDiameter, bilateralSigmaColor, bilateralSigmaSpace)
        #fast = cv2.fastNlMeansDenoising(img, None, nonLocalMeansH, nonLocalMeansKernelDim, nonLocalMeansSearchWindowDim)  

        cv2.waitKey()
        cv2.destroyAllWindows()
        
        print ('----------------------------------------')
        cv2.waitKey()
        cv2.destroyAllWindows()

#@profile
def pixelAverageMask(mask):
    for file in os.listdir('./caps'):
        #t1 = cv2.getTickCount()

        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        magnitude = calcMagnitude(img)
        summa = cv2.sumElems(magnitude[mask])
        num = cv2.countNonZero(magnitude[mask])
        average = np.mean(magnitude[mask])
        average2 = summa[0]/num
        #t2 = cv2.getTickCount()
        #time = (t2 - t1)/ cv2.getTickFrequency()

        cv2.imshow("caps/" + file, magnitude)
        print("caps/" + file + " pixel's magnitude average: " + str(average))
        print("caps/" + file + " pixel's non zero sum: " + str(summa))
        print("caps/" + file + " pixel's non zero count: " + str(num))
        print("caps/" + file + " pixel's magnitude average2: " + str(average2))

        print("caps/" + file + " pixel's average: " + str(np.mean(img[mask])))
        #time = (t2 - t1)/ cv2.getTickFrequency()
        #print ('time: ' + str(time))
        
        print ('----------------------------------------')
        cv2.waitKey()
        cv2.destroyAllWindows()

# called for every image, input: center and radius of inner circle
def pixelAverageGradientMask(img, center, radius):
    if radius is None:
        print('----')
    else:
        magnitude1 = calcMagnitude(img)
        fast = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
        magnitude2 = calcMagnitude(fast)
        edges1 = cv2.Canny(magnitude1, 60, 80, apertureSize=3, L2gradient=True)
        edges2 = cv2.Canny(magnitude2, 60, 80, apertureSize=3, L2gradient=True)

        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        edges3 = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
        cv2.imshow('edges3', edges3)

        # mask = circularmask(576, 768, (center[1], center[0]), radius)
        mask = circularmask(576, 768, (center[1], center[0]), radius-5)

        # cv2.imshow('magnitude1', magnitude1)
        # cv2.imshow('magnitude2', magnitude2)
        # cv2.imshow('edges1', edges1)
        # cv2.imshow('edges2', edges2)
        # histr = cv2.calcHist([edges2[mask]], [0], None, [256], [0,256])
        # plt.plot(histr)
        # plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        average1 = np.mean(magnitude1[mask])
        average2 = np.mean(magnitude2[mask])
        average3 = np.mean(edges1[mask])
        average4 = np.mean(edges2[mask])
        average5 = np.mean(edges3[mask])
        
        print('average1:' + str(average1))
        print('average2:' + str(average2))
        print('average3:' + str(average3))
        print('average4:' + str(average4))
        print('average5:' + str(average5))

        magnitude1 = cv2.cvtColor(magnitude1, cv2.COLOR_GRAY2BGR)
        magnitude2 = cv2.cvtColor(magnitude2, cv2.COLOR_GRAY2BGR)
        edges1 = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)
        edges2 = cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR)

        # cv2.circle(magnitude1, (int(center[1]), int(center[0])), int(radius+1), (0, 255, 0), 1)
        # cv2.circle(magnitude1, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)
        # cv2.circle(magnitude2, (int(center[1]), int(center[0])), int(radius+1), (0, 255, 0), 1)
        # cv2.circle(magnitude2, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)
        # cv2.circle(edges1, (int(center[1]), int(center[0])), int(radius+1), (0, 255, 0), 1)
        # cv2.circle(edges1, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)
        # cv2.circle(edges2, (int(center[1]), int(center[0])), int(radius+1), (0, 255, 0), 1)
        # cv2.circle(edges2, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)

        # cv2.circle(magnitude1, (int(center[1]), int(center[0])), int(radius-5), (0, 255, 0), 1)
        # cv2.circle(magnitude1, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)
        # cv2.circle(magnitude2, (int(center[1]), int(center[0])), int(radius-5), (0, 255, 0), 1)
        # cv2.circle(magnitude2, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)
        # cv2.circle(edges1, (int(center[1]), int(center[0])), int(radius-5), (0, 255, 0), 1)
        # cv2.circle(edges1, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)
        # cv2.circle(edges2, (int(center[1]), int(center[0])), int(radius-5), (0, 255, 0), 1)
        # cv2.circle(edges2, (int(center[1]), int(center[0])), 2, (0, 0, 255), 3)

# call pixelAverageGradientMask
def averageOnAll():
    for file in os.listdir('./caps'):
        print(file)
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('caps/' + file, img)

        # magnitude = calcMagnitude(img)
        # fast = cv2.fastNlMeansDenoising(magnitude, None, 10, 7, 21)
        # cv2.imshow("fast", fast)

        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

        # blobs = labelling.bestLabellingTestGradient(fast)
        blobs = labelling.bestLabellingGradient(edges)

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
            # cv2.imshow('caps/' + file + ' circles', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        pixelAverageGradientMask(img, (x, y), r)

        print('----------------------------------')

def thresholdInner():
    thresh = 0
    prefixed = [filename for filename in os.listdir('./caps') if filename.startswith("g")]
    for file in prefixed:
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

        blobs = labelling.bestLabellingGradient(edges)

        circles = []
        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        mask = circularmask(img.shape[0], img.shape[1], (y, x), r-5)
        avg = np.mean(edges[mask])
        print(file + ' ' + str(avg))

        if avg > thresh:
            thresh = avg

    return thresh

def circularmask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def detect_liner_defects():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5, 5), 2)
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

        blobs = labelling.bestLabellingGradient(edges)

        circles = []
        for blob in blobs:
            if len(blob[0]) > 2:
                x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
                    if r < 210 and r > 170 and x > 0 and y > 0:
                        circles.append((x, y, r, n))

        x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None and y is None and r is None):
            mask = circularmask(img.shape[0], img.shape[1], (y, x), r-10)

            edges = cv2.Canny(gaussian, 20, 100, apertureSize=3, L2gradient=True)

            #image containing only defects
            edges[~mask] = 0
            cv2.imshow("defect", edges)
            
            hasDefects = False
            detected_defect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if contour.size > 100 :
                    hasDefects = True
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    detected_defect = cv2.drawContours(detected_defect, [box], 0, (0,0,255), 1)
            
            if not hasDefects :
                print('caps/' + file + ' has no defects')
            else:
                print('caps/' + file + ' has defects')
                cv2.imshow('detected_defect', detected_defect)  
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

#@profile
def test():
    goodCap = cv2.imread("./caps/g_04.bmp", cv2.IMREAD_GRAYSCALE)
    #incompleteLiner = cv2.imread("./caps/d_18.bmp", cv2.IMREAD_GRAYSCALE)
    #missingLiner = cv2.imread("./caps/d_31.bmp", cv2.IMREAD_GRAYSCALE)
    goodCapMagnitude = calcMagnitude(goodCap)
    #incompleteLinerMagnitude = calcMagnitude(incompleteLiner)
    #missingLinerMagnitude = calcMagnitude(missingLiner)

    #cv2.imshow("Good Cap Magnitude", goodCapMagnitude)
    #cv2.imshow("Incomplete Liner Magnitude", incompleteLinerMagnitude)
    #cv2.imshow("Missing Liner Magnitude", missingLinerMagnitude)

    #print("Good Cap Magnitude average of pixels: " + str(np.mean(goodCapMagnitude)))
    #print("Incomplete Liner Magnitude average of pixels: " + str(np.mean(incompleteLinerMagnitude)))
    #print("Missing Liner Magnitude average of pixels: " + str(np.mean(missingLinerMagnitude)))

    #print("Good Cap average of pixels: " + str(np.mean(goodCap)))
    #print("Incomplete Liner average of pixels: " + str(np.mean(incompleteLiner)))
    #print("Missing Liner average of pixels: " + str(np.mean(missingLiner)))

    #cv2.waitKey()
    #cv2.destroyAllWindows()

    #median = cv2.medianBlur(goodCapMagnitude, 5)
    #gaussian = cv2.GaussianBlur(goodCapMagnitude, (5, 5), sigmaX=0, sigmaY=0)
    #cv2.imshow("gaussian", gaussian)
    #bilateral = cv2.bilateralFilter(goodCapMagnitude, 5, 100, 100)
    #cv2.imshow("bilateral", bilateral)
    fast1 = cv2.fastNlMeansDenoising(goodCapMagnitude, None, 10, 7, 21)
    cv2.imshow("fast1", fast1)

    blobs = labelling.bestLabellingTestGradient(fast1)

    img = cv2.cvtColor(goodCap,cv2.COLOR_GRAY2BGR)
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
        cv2.imshow('caps circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #hist = cv2.calcHist([fast1], [0], None, [256], [0,256])
    #plt.plot(hist)
    #plt.show()

    #ret, thresh = cv2.threshold(fast1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(fast1, 40, 255, cv2.THRESH_BINARY)
    #cv2.imshow("thresh", thresh)
    #print("thresh " + str(ret))

    cv2.waitKey()
    cv2.destroyAllWindows()

    #blobs = labelling.bestLabellingTestGradient(goodCap)

    ##for i in blobs:
    ##    print (i)

    ##for i in range(0, len(blobs)):
    ##    print (i)
    ##    print (blobs[i])

    ##print (len(blobs))

    #img = cv2.cvtColor(goodCap, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('original', img)
    #cv2.waitKey()

    #circles = []

    #for blob in blobs:
    #    #if len(blob[0]) > 1 and checkCollinear(blob[0], blob[1]) == False:
    #    #if len(blob[0]) > 1:
    #    if len(blob[0]) > 2:
    #        x, y, r, n = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
    #        if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
    #            circles.append((x, y, r, n))

    ##print (len(circles))
    #x, y, r = outliers.outliersElimination(circles, (100, 100))
    #cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
    #cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
    #cv2.imshow('circles', img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    #for i in range(0, 10):
    # test()

    #for i in range(0, 20):
    #    mask = circularmask(576, 768, center=None, radius=None)
    #    pixelAverageMask(mask)

    #    pixelAverage()

    # pixelAverage()
    #mask = circularmask(576, 768, center=None, radius=None)
    #pixelAverageMask(mask)

    #averageOnAll()
    detect_liner_defects()