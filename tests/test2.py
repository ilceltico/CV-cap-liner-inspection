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
import circledetection_cook_version

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
            #         if not (math.isnan(x) or math.isnan(y) or math.isnan(r)):
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


def test():
    #thresholdLiner, thresholdDefects = getThresholds()
    thresholdLiner = getThreshold()
    print("thresholdLiner: " + str(thresholdLiner))
    #print("thresholdDefects: " + str(thresholdDefects))
    prefixed = [file for file in os.listdir('./caps') if file.startswith('g_01') or file.startswith('g_06')]
    for file in prefixed:
#    for file in os.listdir('./caps'):
        print("--------------------------------------------------------------------")
        print(file)
        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        binary = binarization.binarize(img)
        #cv2.imshow('binary', binary)

        #TEST IF THE CAP IS A CIRCLE (TASK0 ?)
        if not binarization.is_circle(binary) :
            print('The cap in ' + file + ' is NOT a circle')
            continue
        else:
            print('The cap in ' + file + ' is a circle')

        #if we use directly binary as mask we obtain an image with a line in the middle, here's a test
        #temp = img.copy()
        #temp[~binary] = 0
        #cv2.imshow('temp', temp)

        #we need to convert it to a boolean mask (as linerdefects_gradient.circularmask does: it creates a circular boolean mask)
        mask = binary.copy().astype(bool)
        
        #TASK1
        print("TASK1")
        # outline the cap
        edges = cv2.Canny(binary, 100, 200, apertureSize=3, L2gradient=True)
        
        blobs = labelling.bestLabellingGradient(edges)

        imgOuter = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = []

        for blob in blobs:
                x, y, r = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not (math.isnan(x) or math.isnan(y) or math.isnan(r)):
                    circles.append((x, y, r, len(blob[0])))
                    # circles.append((x, y, r, len(blob[0]), blob))

        # circles_remaining = outliers.outliersElimination_test(circles, (20,20))

        # blob_x = [x for circle in circles_remaining for x in circle[4][0]]
        # blob_y = [y for circle in circles_remaining for y in circle[4][1]]

        # x, y, rCap, cook_d = circledetection_cook_version.leastSquaresCookVersion(blob_x, blob_y)

        # blob_x = [x for _, x in sorted(zip(cook_d, blob_x), reverse=True)]
        # blob_y = [y for _, y in sorted(zip(cook_d, blob_y), reverse=True)]

        # blob_x = blob_x[int(len(blob_x) / 20):]
        # blob_y = blob_y[int(len(blob_y) / 20):]

        # x, y, rCap, cook_d = circledetection_cook_version.leastSquaresCookVersion(blob_x, blob_y)


        

        x, y, rCap = outliers.outliersElimination(circles, (20, 20))
        if not (x is None or y is None or rCap is None):
            cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(rCap).astype("int"), (0, 255, 0), 1)
            cv2.circle(imgOuter, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' outer circle (cap)', imgOuter)
            
            # print position of the center of the cap, diameter of the cap and answer to: is the liner missing - is the liner incomplete?
            print("Position of the center of the cap: (" + str(x) + ", " + str(y) + ")")
            print("Diameter of the cap: " + str(2*rCap))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("Is the liner missing? ")
            #mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (y, x), rCap)
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
        print(np.min(img))
        print(np.min(img[mask]))
        #cv2.imshow('stretched', stretched)

        #without sharpening
        #critic images with bad liner detection: g_01
        gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
        #cv2.imshow('gaussian', gaussian)

        #bilateral = cv2.bilateralFilter(gaussian, 3, 45, 45)
        #cv2.imshow('bilateral', bilateral)

        #fast = cv2.fastNlMeansDenoising(gaussian, None, 3, 7, 7)
        #cv2.imshow('fast', fast)

        #Sharpening ===> NEED TO CHANGE AT LINE 964 with r < rCap - 50 because sharpening enhance part of the image (see edge detection output (line 951))

        #approach1. Critic images with bad liner detection: d_18
        #gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
        #sharpened = cv2.addWeighted(stretched, 1.5, gaussian, -0.5, 0)
        ##cv2.imshow("sharpened", sharpened)
        #gaussian = cv2.GaussianBlur(sharpened, (7,7), 2, 2)

        #approach2: convolution with a high-pass filter
        #critic images with bad liner detection: almost all, some very very bad result with false defect detection, very wrong liner
        #kernel = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]]) 
        #sharpened = cv2.filter2D(stretched, -1, kernel)
        ##cv2.imshow("sharpened", sharpened)
        #gaussian = cv2.GaussianBlur(sharpened, (9,9), 2, 2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #TASK2
        print("TASK2")
        # outline the liner
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        blobs = labelling.bestLabellingGradient(edges)

        imgInner = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = []

        #Blob division for outliersElimination
        # for blob in blobs:
        #         blobX = blob[0]
        #         blobY = blob[1]
        #         length = len(blobX)
        #         length = int(length/5)+1
        #         for i in range(0,10):
        #             imgProva = imgInner.copy()
        #             color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        #             for j in range(0, length):
        #                 if j+i*length < len(blobX):
        #                     imgProva[blobX[j+i*length]][blobY[j+i*length]] = color

        #             if length-1 + i*length >= len(blobX):
        #             	maxIndex = len(blobX)-1
        #             else:
        #             	maxIndex = length-1 + i*length
        #             x, y, r = circledetection.leastSquaresCircleFitCached(blobX[i*length:maxIndex+1], blobY[i*length:maxIndex+1])
        #             if not (math.isnan(x) or math.isnan(y) or math.isnan(r)):
        #             #if r < rCap - 5 and r > 150:
        #                 if r < 0.99*rCap:
        #                     circles.append((x, y, r, len(blobX[i*length:maxIndex])))
                    
        #                     cv2.circle(imgProva, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r).astype("int"), (0, 255, 0), 1)
        #                     cv2.circle(imgProva, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
        #                     cv2.imshow('prova', imgProva)
        #                     cv2.waitKey(0)
        #                     cv2.destroyAllWindows()

        # print(circles)
        # for circle in circles:
        #     cv2.circle(imgInner, (np.round(circle[1]).astype("int"), np.round(circle[0]).astype("int")), np.round(circle[2]).astype("int"), (0, 255, 0), 1)
        #     cv2.circle(imgInner, (np.round(circle[1]).astype("int"), np.round(circle[0]).astype("int")), 2, (0, 0, 255), 3)

        for blob in blobs:
                x, y, r = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
                if not (math.isnan(x) or math.isnan(y) or math.isnan(r)):
                    #if r < rCap - 5 and r > 150:
                    if r < 0.99*rCap:
                        circles.append((x, y, r, len(blob[0]), blob))

        circles_remaining = outliers.outliersElimination_test(circles, (20,20))

        blob_x = [x for circle in circles_remaining for x in circle[4][0]]
        blob_y = [y for circle in circles_remaining for y in circle[4][1]]

        x, y, r, cook_d = circledetection_cook_version.leastSquaresCookVersion(blob_x, blob_y)

        blob_x = [x for _, x in sorted(zip(cook_d, blob_x), reverse=True)]
        blob_y = [y for _, y in sorted(zip(cook_d, blob_y), reverse=True)]

        normalizedCooks = np.array(cook_d)/max(cook_d) * 200 + 55

        imgCook = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(0, len(blob_x)):
            imgCook[blob_x[i],blob_y[i]] = (normalizedCooks[i],100,0)

        cv2.imshow('imgCook', imgCook)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        eliminationFraction = 10

        blob_x = blob_x[int(len(blob_x) / eliminationFraction):]
        blob_y = blob_y[int(len(blob_y) / eliminationFraction):]

        cook_d = cook_d[int(len(cook_d) / eliminationFraction):]

        normalizedCooks = np.array(cook_d)/max(cook_d) * 200 + 55
        imgCook = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(0, len(blob_x)):
            imgCook[blob_x[i],blob_y[i]] = (normalizedCooks[i],100,0)

        cv2.imshow('imgCook', imgCook)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        x, y, r = circledetection.leastSquaresCircleFitCached(blob_x, blob_y)




        # x, y, r = outliers.outliersElimination(circles, (20, 20))
        if not (x is None or y is None or r is None):
            cv2.circle(imgInner, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(r).astype("int"), (0, 255, 0), 1)
            cv2.circle(imgInner, (np.round(y).astype("int"), np.round(x).astype("int")), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' inner circle (liner)', imgInner)

            # print position of the center of the liner, diameter of the liner
            print("Position of the center of the liner: (" + str(x) + ", " + str(y) + ")")
            print("Diameter of the liner: " + str(2*r))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #DEFECT DETECTION
            print("Is the liner incomplete?")
            mask = linerdefects_gradient.circularmask(img.shape[0], img.shape[1], (y, x), 0.95*r)
            
            #   we can use a pixel average to detect defects and check if it is greater than a threshold

            #avg = np.mean(edges[mask])
            ##print('avg:' + str(avg))

            #if avg > thresholdDefects:
            #    print("caps/" + file + " has defects!")
            #else:
            #    print("caps/" + file + " has no defects!")

            #   or we can check if there are blobs (sufficiently large) in the inner circle (need to perform another edge detection that capture more defect if present)
            
            edges = cv2.Canny(gaussian, 20, 110, apertureSize=3, L2gradient=True)

            #image containing only defects
            edges[~mask] = 0
            #cv2.imshow("defect", edges)
            liner = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            cv2.circle(liner, (np.round(y).astype("int"), np.round(x).astype("int")), np.round(0.95*r).astype("int"), (255, 255, 255), 2)
            #cv2.imshow("liner", liner)
            #liner_defects = liner + edges
            #cv2.imshow("liner+defects", liner_defects)
            nonzero = np.nonzero(liner)
            liner = list(zip(nonzero[0],nonzero[1]))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            hasDefects = False
            detected_defect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            blobs = labelling.bestLabellingGradient(edges)
            for blob in blobs:
                temp = list(zip(blob[0],blob[1]))
                common = list(set(liner).intersection(temp))
                max_distance = 0
                for pixel in common:
                    for pixel2 in common:
                        distance = math.sqrt((pixel[0]-pixel2[0])**2 + (pixel[1]-pixel2[1])**2)
                        if distance > max_distance:
                            max_distance = distance

                if len(common) >= 2 and max_distance > r/10:
                    hasDefects = True
                    rect = cv2.minAreaRect(np.array(list(zip(blob[1], blob[0]))))
                    rectDim = rect[1]
                    #Increase the smaller dimension of the rect, to make it more visible.
                    if rectDim[0] < rectDim[1]:
                        rectDim = (rectDim[0]*2, rectDim[1]*1)
                    else:
                    	rectDim = (rectDim[0]*1, rectDim[1]*2)
                    rect = (rect[0], rectDim, rect[2])
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

            # # dilation to make the defect more evident
            # kernel = np.ones((3,3),np.uint8)
            # edges = cv2.dilate(edges, kernel, iterations=1)
            # cv2.imshow("defect", edges)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
           
            # hasDefects = False
            # detected_defect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # #contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # # contours2, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # # for i in range(0, len(contours)):
            # # 	print(contours[i].size)
            # # 	print(contours2[i].size)
            # # 	print(cv2.contourArea(contours[i]))
            # # 	print(cv2.contourArea(contours2[i]))

            # for contour in contours:
            #     if contour.size > 100 :
            #         hasDefects = True
            #         rect = cv2.minAreaRect(contour)
            #         box = cv2.boxPoints(rect)
            #         box = np.int0(box)
            #         detected_defect = cv2.drawContours(detected_defect, [box], 0, (0,0,255), 1)
                
            # if not hasDefects :
            #     print('caps/' + file + ' has no defects')
            # else:
            #     print('caps/' + file + ' has defects')
            #     cv2.imshow('caps/' + file + ' detected defects', detected_defect)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

if __name__ == '__main__':
    test()