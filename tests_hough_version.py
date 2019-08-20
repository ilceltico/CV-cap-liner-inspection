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

def test_outer_with_binarization():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        binary = binarization.binarize(img)
        #cv2.imshow('caps/' + file + ' binary', binary)

        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
            
        cv2.imshow('./caps/' + file + ': detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def outer_circle_with_stretching():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        #LINEAR STRETCHING
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)

        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)

        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
            
        cv2.imshow('./caps/' + file + ': detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_outer_circle_with_contours():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        binary = binarization.binarize(img)
        # Contours are simply a curve joining all the points (along the boundary), having same color or intensity.
        #first argument: image
        #second argument: contour retrieval mode.
        #third argument: contour approximation method. Two possibilities: 
        #   cv2.CHAIN_APPROX_NONE: all the boundary points are stored.
        #   cv2.CHAIN_APPROX_SIMPLE: removes all redundant points and compresses the contour, saving memory. E.g. for a line store only the two end points.
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img = cv2.drawContours(img, [cnt], 0, (0,255,0), 1)

        #center of the circle

        M = cv2.moments(cnt)
        #baricenter (centroids). Can be computed as m10 / m00 and m01 / m00, knowing that m10 and m01 are sum(i) and sum(j), and m00 is the area.
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        print("center of the circle: (" + str(cy) + ", " + str(cx) + ")")

        pointX = cnt[0][0][0]
        pointY = cnt[0][0][1]
        radius = math.sqrt((pointX - cx)**2 + (pointY - cy)**2)
        print("radius: " + str(radius))
        cv2.circle(img, (np.round(cx).astype("int"), np.round(cy).astype("int")), 2, (0, 0, 255), 3)

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

        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]

        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), circle[2])
        avg = np.mean(img[mask])
        #temp = img.copy()
        #temp[~mask] = 0
        #cv2.imshow("temp", temp)

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
        
        #binary = binarization.binarize(img)
        #edges = cv2.Canny(binary, 45, 100, apertureSize=3, L2gradient=True)

        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]

        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), circle[2])
        avg = np.mean(img[mask])
        #temp = img.copy()
        #temp[~mask] = 0
        #cv2.imshow("temp", temp)

        thresh = thresh + avg

    thresh = thresh / len(prefixed)
    
    return thresh + thresh/10    #to consider the cap with no liner it must have a big difference with the correct average

def getThresholds():
    thresholdLiner = 0
    thresholdDefects = 0
    prefixed = [filename for filename in os.listdir('./caps') if filename.startswith("g")]
    for file in prefixed:
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        
        #binary = binarization.binarize(img)
        #edges = cv2.Canny(binary, 45, 100, apertureSize=3, L2gradient=True)

        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]

        rCap = circle[2]
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), rCap)
        avg = np.mean(img[mask])
        #temp = img.copy()
        #temp[~mask] = 0
        #cv2.imshow("temp", temp)

        thresholdLiner = thresholdLiner + avg

        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=170, maxRadius=rCap-5)
        circles = np.uint16(np.around(circles))
        #first (better) circle
        circle = circles[0][0]

        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), circle[2]-10)
        avg = np.mean(edges[mask])
        #print(file + ' ' + str(avg))
        
        if avg > thresholdDefects:
            thresholdDefects = avg

    thresholdLiner = thresholdLiner / len(prefixed)

    #to consider the cap with no liner or with defects, it must have a big difference wrt thresholds
    return thresholdLiner + thresholdLiner/10, thresholdDefects + thresholdDefects/10

def test_inner_liner_magnitude():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('caps/' + file, img)

        magnitude = linerdefects_gradient.calcMagnitude(img)
        # cv2.imshow("magnitude", magnitude)
        fast = cv2.fastNlMeansDenoising(magnitude, None, 10, 7, 21)
        # cv2.imshow("fast", fast)

        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(fast, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=170, maxRadius=210)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
            
        cv2.imshow('./caps/' + file + ': detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def another_inner_circle():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('caps/' + file, img)

        fast = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
        #cv2.imshow("fast", fast)

        magnitude = linerdefects_gradient.calcMagnitude(fast)
        #cv2.imshow("magnitude", magnitude)

        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(magnitude, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=170, maxRadius=210)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
            
        cv2.imshow('./caps/' + file + ': detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#@profile
def best_inner_circle():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        #LINEAR STRETCHING
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)

        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=170, maxRadius=210)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
            
        cv2.imshow('./caps/' + file + ': detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_sharpening():
    thresholdLiner = getThreshold()
    print("thresholdLiner: " + str(thresholdLiner))
    for file in os.listdir('./caps'):
        print("--------------------------------------------------------------------")
        print(file)
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        #TEST IF THE CAP IS A CIRCLE (TASK0 ?)
        if not binarization.is_circle(img) :
            print('the cap in ' + file + ' is not a circle')
            continue
        else:
            print('The cap in ' + file + ' is a circle')

        #LINEAR STRETCHING and GAUSSIAN FILTERING
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)

        #approach1
        #gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
        #sharpened = cv2.addWeighted(imgOut, 1.5, gaussian, -0.5, 0)
        ##cv2.imshow("sharpened", sharpened)
        #gaussian = cv2.GaussianBlur(sharpened, (5,5), 2, 2)

        #approach2: convolution with a high-pass filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]]) 
        sharpened = cv2.filter2D(imgOut, -1, kernel)
        #cv2.imshow("sharpened", sharpened)
        gaussian = cv2.GaussianBlur(sharpened, (9,9), 2, 2)

        #TASK1
        print("TASK1")
        # outline the cap
        imgOuter = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        rCap = circle[2]
        # draw the outer circle
        cv2.circle(imgOuter,(circle[0],circle[1]),rCap,(0,255,0),1)
        # draw the center of the circle
        cv2.circle(imgOuter,(circle[0],circle[1]),2,(0,0,255),3)

        cv2.imshow('caps/' + file + ' outer circle (cap)', imgOuter)
            
        # print position of the center of the cap, diameter of the cap and answer to: is the liner missing - is the liner incomplete?
        print("Position of the center of the cap: (" + str(circle[1]) + ", " + str(circle[0]) + ")")
        print("Diameter of the cap: " + str(2*rCap))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Is the liner missing? ")
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), rCap)
        avg = np.mean(img[mask])
        print("caps/" + file + " pixel's average: " + str(avg))
        if avg > thresholdLiner:
            print("caps/" + file + " has no liner!!")
            continue
        else:
            print("caps/" + file + " has liner")

        #TASK2
        print("TASK2")
        # outline the liner
        imgInner = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #   maxRadius should be (like in tests.py) rCap-5. But it is a too large value with HoughCircles (images d_20, g_01, g_06 has problems).
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=150, maxRadius=rCap-50)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(imgInner,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(imgInner,(circle[0],circle[1]),2,(0,0,255),3)

        cv2.imshow('caps/' + file + ' inner circle (liner)', imgInner)

        # print position of the center of the liner, diameter of the liner
        print("Position of the center of the liner: (" + str(circle[1]) + ", " + str(circle[0]) + ")")
        print("Diameter of the liner: " + str(2*circle[2]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #DEFECT DETECTION
        print("Is the liner incomplete?")
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), circle[2]-10)
            
        #   we can use a pixel average to detect defects and check if it is greater than a threshold

        #avg = np.mean(edges[mask])
        ##print('avg:' + str(avg))

        #if avg > thresholdDefects:
        #    print("caps/" + file + " has defects!")
        #else:
        #    print("caps/" + file + " has no defects!")

        #   or we can check if there are blobs (sufficiently large) in the inner circle (need to perform another edge detection that capture more defect if present)
            
        edges = cv2.Canny(gaussian, 20, 100, apertureSize=3, L2gradient=True)
        #image containing only defects
        edges[~mask] = 0
        #cv2.imshow("defect", edges)

        # dilation to make the defect more evident
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        #cv2.imshow("defect", edges)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
           
        hasDefects = False
        detected_defect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size > 200 :
                hasDefects = True
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                detected_defect = cv2.drawContours(detected_defect, [box], 0, (0,0,255), 1)
                
        if not hasDefects :
            print('caps/' + file + ' has no defects')
        else:
            print('caps/' + file + ' has defects')
            cv2.imshow('caps/' + file + ' detected defects', detected_defect)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def test_all():
    #thresholdLiner, thresholdDefects = getThresholds()
    thresholdLiner = getThreshold()
    print("thresholdLiner: " + str(thresholdLiner))
    #print("thresholdDefects: " + str(thresholdDefects))
    for file in os.listdir('./caps'):
        print("--------------------------------------------------------------------")
        print(file)
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        #TEST IF THE CAP IS A CIRCLE (TASK0 ?)
        if not binarization.is_circle(img) :
            print('the cap in ' + file + ' is not a circle')
            continue
        else:
            print('The cap in ' + file + ' is a circle')

        #LINEAR STRETCHING and GAUSSIAN FILTERING
        imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
        gaussian = cv2.GaussianBlur(imgOut, (5,5), 2, 2)

        #TASK1
        print("TASK1")
        # outline the cap
        imgOuter = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        rCap = circle[2]
        # draw the outer circle
        cv2.circle(imgOuter,(circle[0],circle[1]),rCap,(0,255,0),1)
        # draw the center of the circle
        cv2.circle(imgOuter,(circle[0],circle[1]),2,(0,0,255),3)

        cv2.imshow('caps/' + file + ' outer circle (cap)', imgOuter)
            
        # print position of the center of the cap, diameter of the cap and answer to: is the liner missing - is the liner incomplete?
        print("Position of the center of the cap: (" + str(circle[1]) + ", " + str(circle[0]) + ")")
        print("Diameter of the cap: " + str(2*rCap))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Is the liner missing? ")
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), rCap)
        avg = np.mean(img[mask])
        print("caps/" + file + " pixel's average: " + str(avg))
        if avg > thresholdLiner:
            print("caps/" + file + " has no liner!!")
            continue
        else:
            print("caps/" + file + " has liner")

        #TASK2
        print("TASK2")
        # outline the liner
        imgInner = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #   maxRadius should be (like in tests.py) rCap-5. But it is a too large value with HoughCircles (images d_20, g_01, g_06 has problems).
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=150, maxRadius=rCap-50)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(imgInner,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(imgInner,(circle[0],circle[1]),2,(0,0,255),3)

        cv2.imshow('caps/' + file + ' inner circle (liner)', imgInner)

        # print position of the center of the liner, diameter of the liner
        print("Position of the center of the liner: (" + str(circle[1]) + ", " + str(circle[0]) + ")")
        print("Diameter of the liner: " + str(2*circle[2]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #DEFECT DETECTION
        print("Is the liner incomplete?")
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), circle[2]-10)

        #   we can use a pixel average to detect defects and check if it is greater than a threshold

        #avg = np.mean(edges[mask])
        ##print('avg:' + str(avg))

        #if avg > thresholdDefects:
        #    print("caps/" + file + " has defects!")
        #else:
        #    print("caps/" + file + " has no defects!")

        #   or we can check if there are blobs (sufficiently large) in the inner circle (need to perform another edge detection that capture more defect if present)

        edges = cv2.Canny(gaussian, 20, 100, apertureSize=3, L2gradient=True)
        #image containing only defects
        edges[~mask] = 0
        #cv2.imshow("defect", edges)

        # dilation to make the defect more evident
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        #cv2.imshow("defect", edges)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
           
        hasDefects = False
        detected_defect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.imshow('caps/' + file + ' detected defects', detected_defect)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def test():
    #thresholdLiner, thresholdDefects = getThresholds()
    thresholdLiner = getThreshold()
    print("thresholdLiner: " + str(thresholdLiner))
    #print("thresholdDefects: " + str(thresholdDefects))
    for file in os.listdir('./caps'):
        print("--------------------------------------------------------------------")
        print(file)
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        #TEST IF THE CAP IS A CIRCLE (TASK0 ?)
        if not binarization.is_circle(img) :
            print('the cap in ' + file + ' is not a circle')
            continue
        else:
            print('The cap in ' + file + ' is a circle')

        binary = binarization.binarize(img)
        #cv2.imshow('binary', binary)

        #if we use directly binary as mask we obtain an image with a line in the middle, here a test
        #temp = img.copy()
        #temp[~binary] = 0
        #cv2.imshow('temp', temp)

        #we need to convert it to a boolean mask (as linerdefects_gradient.circularmask does: it create a circular boolean mask)
        mask = binary.copy().astype(bool)
        
        #TASK1
        print("TASK1")
        # outline the cap
        imgOuter = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        rCap = circle[2]
        # draw the outer circle
        cv2.circle(imgOuter,(circle[0],circle[1]),rCap,(0,255,0),1)
        # draw the center of the circle
        cv2.circle(imgOuter,(circle[0],circle[1]),2,(0,0,255),3)

        cv2.imshow('caps/' + file + ' outer circle (cap)', imgOuter)
            
        # print position of the center of the cap, diameter of the cap and answer to: is the liner missing - is the liner incomplete?
        print("Position of the center of the cap: (" + str(circle[1]) + ", " + str(circle[0]) + ")")
        print("Diameter of the cap: " + str(2*rCap))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Is the liner missing? ")
        #mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), rCap)
        avg = np.mean(img[mask])
        print("caps/" + file + " pixel's average: " + str(avg))
        if avg > thresholdLiner:
            print("caps/" + file + " has no liner!!")
            continue
        else:
            print("caps/" + file + " has liner")

        #LINEAR STRETCHING and GAUSSIAN FILTERING
        #linear stretching only on the mask (cap)
        stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
        stretched[~mask] = 0
        #cv2.imshow('stretched', stretched)

        #without sharpening
        #critic images with bad liner detection: d_18, d_20
        gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
        #cv2.imshow('gaussian', gaussian)
        #edges = cv2.Canny(gaussian, 50, 100)
        #cv2.imshow('edges', edges)

        #Sharpening
        #approach1. Critic images with bad liner detection: d_18, d_20
        #gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
        #sharpened = cv2.addWeighted(stretched, 1.5, gaussian, -0.5, 0)
        ##cv2.imshow("sharpened", sharpened)
        #gaussian = cv2.GaussianBlur(sharpened, (7,7), 2, 2)
        #edges = cv2.Canny(gaussian, 50, 100)
        #cv2.imshow('edges', edges)

        #approach2: convolution with a high-pass filter
        #critic images with bad liner detection: almost all, some very very bad result with false defect detection, very wrong liner
        #kernel = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]]) 
        #sharpened = cv2.filter2D(stretched, -1, kernel)
        ##cv2.imshow("sharpened", sharpened)
        #gaussian = cv2.GaussianBlur(sharpened, (9,9), 2, 2)
        #edges = cv2.Canny(gaussian, 50, 100)
        #cv2.imshow('edges', edges)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #TASK2
        print("TASK2")
        # outline the liner
        imgInner = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #   maxRadius should be (like in tests.py) rCap-5. But it is a too large value with HoughCircles (images d_20, g_01, g_06 has problems).
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=150, maxRadius=rCap-50)

        circles = np.uint16(np.around(circles))
        #draw only the first (better) circle
        circle = circles[0][0]
        # draw the outer circle
        cv2.circle(imgInner,(circle[0],circle[1]),circle[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(imgInner,(circle[0],circle[1]),2,(0,0,255),3)

        cv2.imshow('caps/' + file + ' inner circle (liner)', imgInner)

        # print position of the center of the liner, diameter of the liner
        print("Position of the center of the liner: (" + str(circle[1]) + ", " + str(circle[0]) + ")")
        print("Diameter of the liner: " + str(2*circle[2]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #DEFECT DETECTION
        print("Is the liner incomplete?")
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), circle[2]-15)

        #   we can use a pixel average to detect defects and check if it is greater than a threshold

        #avg = np.mean(edges[mask])
        ##print('avg:' + str(avg))

        #if avg > thresholdDefects:
        #    print("caps/" + file + " has defects!")
        #else:
        #    print("caps/" + file + " has no defects!")

        #   or we can check if there are blobs (sufficiently large) in the inner circle (need to perform another edge detection that capture more defect if present)

        edges = cv2.Canny(gaussian, 20, 100, apertureSize=3, L2gradient=True)
        #image containing only defects
        edges[~mask] = 0
        cv2.imshow("defect", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # dilation to make the defect more evident
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        #cv2.imshow("defect", edges)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
           
        hasDefects = False
        detected_defect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.imshow('caps/' + file + ' detected defects', detected_defect)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    #test_outer_with_binarization()
    #test_missing_liner()
    #test_inner_liner_magnitude()
    #another_inner_circle()
    #best_inner_circle()
    #outer_circle_with_stretching()
    #test_outer_circle_with_erosion()
    #test_outer_circle_with_dilation()
    #test_outer_circle_with_contours()

    #test_sharpening()
    #test_all()
    test()