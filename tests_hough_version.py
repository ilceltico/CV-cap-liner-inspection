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
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=5, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))

        #   draw only the first (better) circle
        #circle = circles[0][0]
        ## draw the outer circle
        #cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        ## draw the center of the circle
        #cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
        #cv2.imshow('./caps/' + file + ': detected circle', cimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #   draw a circle interpolated from the best 3 circles found with Hough
        #c = []
        #for i in range(0, 3):
        #    circle = circles[0][i]
        #    # draw the outer circle
        #    #cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),1)
        #    # draw the center of the circle
        #    #cv2.circle(cimg,(circle[0],circle[1]),2,(0,0,255),3)
        #    #   we don't have the number of pixel in the circle, we approximate it with the circumference (?)
        #    #print(str(np.round(2*math.pi*circle[2]).astype("int")))
        #    c.append((circle[1], circle[0], circle[2], np.round(2*math.pi*circle[2]).astype("int")))

        #x, y, r = outliers.outliersElimination(c, (20, 20))
        #if not (x is None and y is None and r is None):
        #    cv2.circle(cimg, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r).astype("int"), (0, 255, 0), 1)
        #    cv2.circle(cimg, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        #    cv2.imshow('caps/' + file + ' circles', cimg)
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()

        #   get the only center of the best circle with Hough (or the mean of the best three)
        #circle = circles[0][0]
        #y = circle[0]
        #x = circle[1]

        c = []
        sum_x = 0
        sum_y = 0
        for i in range(0, 3):
            x = circles[0][i][1]
            y = circles[0][i][0]
            sum_x += x
            sum_y += y

        x = sum_x/3
        y = sum_y/3

        #compute the radius as the mean distance between points and the center (found with Hough)
        edges = cv2.Canny(binary, 100, 200, L2gradient=True)
        pixels_x, pixels_y = np.nonzero(edges)

        radius = np.sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
        print(radius)
        cv2.circle(cimg, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(radius).astype("int"), (0, 255, 0), 1)
        cv2.circle(cimg, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': detected circle', cimg)
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
    #First element is the average of perfect caps, second element of missing liners
    average = [0,0]
    i = 0
    for fileStart in ['g', 'd_31']:
        prefixed = [filename for filename in os.listdir('./caps') if filename.startswith(fileStart)]
        for file in prefixed:
            img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        
            #binary = binarization.binarize(img)
            #edges = cv2.Canny(binary, 45, 100, apertureSize=3, L2gradient=True)

            # imgOut = ((255 / (img.max() - img.min()))*(img.astype(np.float)-img.min())).astype(np.uint8)
            # gaussian = cv2.GaussianBlur(imgOut, (5,5), 2)
            # edges = cv2.Canny(gaussian, 100, 200, apertureSize=3, L2gradient=True)
            # blobs = labelling.bestLabellingGradient(edges)

            # circles = []
            # for blob in blobs:
            #     if len(blob[0]) > 2:
            #         x, y, r = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
            #         if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
            #            circles.append((x, y, r, len(blob[0])))

            # x, y, r = outliers.outliersElimination(circles, (20, 20))
            # mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (y, x), r)
            # avg = np.mean(img[mask])
            # temp = img.copy()
            # temp[~mask] = 255
            # cv2.imshow("leastCircles Mask", temp)

            binary = binarization.binarize(img)
            mask = binary.copy().astype(bool)
            avg = np.mean(img[mask])
            #temp2 = img.copy()
            #temp2[~mask] = 255
            #cv2.imshow("binarization Mask", temp2)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            average[i] = average[i] + avg
    
        average[i] = average[i] / len(prefixed)

        i = i+1

    #print('Good caps average: ' + str(average[0]))
    #print('Missing liners average: ' + str(average[1]))

    thresh = (average[0] + average[1])/2
    #print('Threshold: ' + str(thresh))

    return thresh
    #return thresh + thresh/10    #to consider the cap with no liner it must have a big difference with the correct average

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

        binary = binarization.binarize(img)
        mask = binary.copy().astype(bool)
        
        # outline the cap
        imgOuter = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        
        #   get the only center of the best circle with Hough (or the mean of the best three)
        #circle = circles[0][0]
        #y = circle[0]
        #x = circle[1]

        sum_x = 0
        sum_y = 0
        for i in range(0, 3):
            x = circles[0][i][1]
            y = circles[0][i][0]
            sum_x += x
            sum_y += y

        x = sum_x/3
        y = sum_y/3

        #compute the radius as the mean distance between points and the center (found with Hough)
        edges = cv2.Canny(binary, 100, 200)
        pixels_x, pixels_y = np.nonzero(edges)

        r_cap = np.sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
        cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r_cap).astype("int"), (0, 255, 0), 1)
        cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': detected circle (cap)', imgOuter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #LINEAR STRETCHING and GAUSSIAN FILTERING
        #linear stretching only on the mask (cap)
        stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
        stretched[~mask] = 0
        gaussian = cv2.GaussianBlur(stretched, (9,9), 2, 2)
        #cv2.imshow('caps/' + file + ' gaussian', gaussian)
        edges = cv2.Canny(gaussian, 50, 100)
        #cv2.imshow('caps/' + file + ' edges', edges)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        imgInner = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=0, maxRadius=np.round(0.9*r_cap).astype("int"))
        circles = np.uint16(np.around(circles))
        x = circles[0][0][1]
        y = circles[0][0][0]
        r_liner = circles[0][0][2]
                
        cv2.circle(imgInner, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r_liner).astype("int"), (0, 255, 0), 1)
        cv2.circle(imgInner, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': detected circle (liner) HOUGH', imgInner)

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
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=5, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        
        #   draw only the first (better) circle
        #circle = circles[0][0]
        #rCap = circle[2]
        ## draw the outer circle
        #cv2.circle(imgOuter,(circle[0],circle[1]),rCap,(0,255,0),1)
        ## draw the center of the circle
        #cv2.circle(imgOuter,(circle[0],circle[1]),2,(0,0,255),3)
        #cv2.imshow('caps/' + file + ' inner circle (liner)', imgOuter)

        #   take only the radius from the Hough result and compute the radius
        c = []
        sum_x = 0
        sum_y = 0
        for i in range(0, 3):
            x = circles[0][i][1]
            y = circles[0][i][0]
            sum_x += x
            sum_y += y

        x = sum_x/3
        y = sum_y/3

        #compute the radius as the mean distance between points and the center (found with Hough)
        edges = cv2.Canny(binary, 100, 200, L2gradient=True)
        pixels_x, pixels_y = np.nonzero(edges)

        r_cap = np.sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
        cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r_cap).astype("int"), (0, 255, 0), 1)
        cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': outer circle (cap)', imgOuter)
            
        # print position of the center of the cap, diameter of the cap and answer to: is the liner missing - is the liner incomplete?
        print("Position of the center of the cap: (" + str(x) + ", " + str(y) + ")")
        print("Diameter of the cap: " + str(2*r_cap))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Is the liner missing? ")
        #mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), r_cap)
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
        gaussian = cv2.GaussianBlur(stretched, (9,9), 2, 2)
        #cv2.imshow('gaussian', gaussian)
        #edges = cv2.Canny(gaussian, 50, 100)
        #cv2.imshow('edges', edges)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #TASK2
        print("TASK2")
        # outline the liner
        imgInner = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=0, maxRadius=np.round(0.9*r_cap).astype("int"))

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
        mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (circle[0], circle[1]), 0.95*circle[2])

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

def compare_inner():
    for file in os.listdir('./caps'):
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        binary = binarization.binarize(img)
        mask = binary.copy().astype(bool)
        
        # outline the cap
        imgOuter = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=10, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        
        #   get the only center of the best circle with Hough (or the mean of the best three)
        #circle = circles[0][0]
        #y = circle[0]
        #x = circle[1]

        sum_x = 0
        sum_y = 0
        for i in range(0, 3):
            x = circles[0][i][1]
            y = circles[0][i][0]
            sum_x += x
            sum_y += y

        x = sum_x/3
        y = sum_y/3

        #compute the radius as the mean distance between points and the center (found with Hough)
        edges = cv2.Canny(binary, 100, 200)
        pixels_x, pixels_y = np.nonzero(edges)

        r_cap = np.sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
        cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r_cap).astype("int"), (0, 255, 0), 1)
        cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': detected circle (cap)', imgOuter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #LINEAR STRETCHING and GAUSSIAN FILTERING
        #linear stretching only on the mask (cap)
        stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
        stretched[~mask] = 0
        gaussian = cv2.GaussianBlur(stretched, (9,9), 2, 2)
        
        imgInner = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=0, maxRadius=np.round(0.9*r_cap).astype("int"))
        circles = np.uint16(np.around(circles))

        out_mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (y, x), 0.9*r_cap)

        x = circles[0][0][1]
        y = circles[0][0][0]
        r_liner = circles[0][0][2]
        print('HOUGH: ' + str(r_liner))
        print('HOUGH: ' + str(x))
        print('HOUGH: ' + str(y))        
        cv2.circle(imgInner, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r_liner).astype("int"), (0, 255, 0), 1)
        cv2.circle(imgInner, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': detected circle (liner) HOUGH', imgInner)
        
        sum_x = 0
        sum_y = 0
        sum_r = 0
        for i in range(0, 3):
            x = circles[0][i][1]
            y = circles[0][i][0]
            r = circles[0][i][2]
            sum_x += x
            sum_y += y
            sum_r += r

        x = sum_x/3
        y = sum_y/3
        r = sum_r/3
        in_mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (y, x), 0.95*r)
        gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
        edges = cv2.Canny(gaussian, 45, 100, L2gradient=True)
        edges[~out_mask] = 0
        edges[in_mask] = 0
        cv2.imshow('liner', edges)

        if len(np.nonzero(edges)[0]) == 0:  #skip d_31
            continue

        pixels_x, pixels_y = np.nonzero(edges)
        r_liner = np.sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
        print('NEW: ' + str(r_liner))
        print('NEW: ' + str(x))
        print('NEW: ' + str(y))
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.circle(cimg, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r_liner).astype("int"), (0, 255, 0), 1)
        cv2.circle(cimg, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        cv2.imshow('./caps/' + file + ': detected circle (liner) NEW METHOD', cimg)

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
    compare_inner()
    #test_sharpening()
    #test_all()
    test()