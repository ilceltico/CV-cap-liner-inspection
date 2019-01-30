import numpy as np
import cv2
import math
import Labelling
import ProjectTest_Gava

#@profile
def outliersElimination(circles, thresholds):
    # circles: list of tuple (center x, center y, radius, number of pixel)
    #thresholds: tuple of thresholds (center, radius)

    weighted = [[x * n, y * n, r * n, n] for x, y , r, n in circles]
    sums = [sum(a) for a in zip(*weighted)]
    finalCircle = [el/sums[3] for el in sums]

    #xAccumulator = 0.0
    #yAccumulator = 0.0
    #rAccumulator = 0.0
    #nAccumulator = 0

    #xAccumulator = sum(x * n for x, _, _, n in circles)
    #yAccumulator = sum(y * n for _, y, _, n in circles)
    #rAccumulator = sum(r * n for _, _ , r, n in circles)
    #nAccumulator = sum(n for _, _, _, n in circles)

    #newX = xAccumulator / nAccumulator
    #newY = yAccumulator / nAccumulator
    #newR = rAccumulator / nAccumulator

    #print (newX)
    #print (newY)
    #print (newR)

    circlesRemaining = []

    for element in circles:
        if math.sqrt((element[0] - finalCircle[0]) ** 2 + (element[1] - finalCircle[1]) ** 2) <= thresholds[0] and abs(element[2] - finalCircle[2]) <= thresholds[1]:
            circlesRemaining.append(element)
    
    #print (len(circles))
    #print (circles)

    weighted2 = [[x * n, y * n, r * n, n] for x, y , r, n in circlesRemaining]
    sums2 = [sum(a) for a in zip(*weighted2)]
    finalCircle2 = [el/sums2[3] for el in sums2]

    #weightedSums2 = [np.sum([x * n, y * n, r * n, n], axis=0) for x, y, r, n in circles]
    #finalCircle2 = [el/weightedSums2[3] for el in weightedSums2]

    return finalCircle2[0], finalCircle2[1], finalCircle2[2]


#circles = [(281.3, 390.7, 204.7, 1890), (276.5, 387.4, 201.6, 1993), (285.9, 382.5, 200.4, 2123), (1234.9, 1435.3, 54.7, 127)]
#thresholds = (10, 10)

#for i in range (0, 1):
#    a = outliersElimination(circles, thresholds)
#    #print (a)

@profile
def test():
    blobs = Labelling.bestLabelling('d_16.bmp')

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
        x, y, r, n = ProjectTest_Gava.leastSquaresCircleFitCached(blob[0], blob[1])
        #print (x)
        #print (y)
        #print (r)

        if not math.isnan(x) or not math.isnan(y) or not math.isnan(r):
            #cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
            #cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
            #cv2.imshow('circles', img)
            #cv2.waitKey()
            circles.append((x, y, r, n))

    print (len(circles))
    x, y, r = outliersElimination(circles, (50, 30))
    #cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
    #cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
    #cv2.imshow('circles', img)
    #cv2.waitKey()


for i in range(0, 10):
    test()
    print (i)