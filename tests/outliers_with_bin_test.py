import numpy as np
import cv2
import math
import labelling
import circledetection

#@profile
def outliersElimination(circles, thresholds):
    # circles: list of tuple (center x, center y, radius, number of pixel)
    #thresholds: tuple of thresholds (center, radius)

    if len(circles) == 0:
        return None, None, None

    weighted = [[x * n, y * n, r * n, n] for x, y , r, n in circles]
    #print('weighted: ' + str(weighted))
    #print('zip(*weighted): ' + str(set(zip(*weighted))))
    sums = [sum(a) for a in zip(*weighted)]
    #print('sums: ' + str(sums))
    meanCircle = [el/sums[3] for el in sums]
    #print('meanCircle: ' + str(meanCircle))

    #print (meanCircle)


    #splitted = np.split(circles, [3], axis=1)
    #values = splitted[0].tolist()
    #weights = [w[0] for w in splitted[1]]

    #res = np.average(values, axis=0, weights=weights)

    #print (res)

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

    #circlesRemaining = []

    #for element in circles:
    #    if math.sqrt((element[0] - meanCircle[0]) ** 2 + (element[1] - meanCircle[1]) ** 2) <= thresholds[0] and abs(element[2] - meanCircle[2]) <= thresholds[1]:
    #        circlesRemaining.append(element)

    #Eliminates circles with a center that is too far from the mean center according to threshold[0] 
    # or a radius too different from the mean radius according to threshold[1]
    circlesRemaining = [(x, y, r, n) for x, y, r, n in circles if math.sqrt((x - meanCircle[0]) ** 2 + (y - meanCircle[1]) ** 2) <= thresholds[0] and abs(r - meanCircle[2]) <= thresholds[1]]
    
    #print (len(circles))
    #print (circles)

    if len(circlesRemaining) > 0:
        weightedRemaining = [[x * n, y * n, r * n, n] for x, y, r, n in circlesRemaining]
        sumsRemaining = [sum(a) for a in zip(*weightedRemaining)]
        finalCircle = [el/sumsRemaining[3] for el in sumsRemaining]

        #splitted2 = np.split(circlesRemaining, [3], axis=1)
        #values2 = splitted2[0].tolist()
        #weightedRemaining = [w[0] for w in splitted2[1]]

        #res2 = np.average(values2, axis=0, weights=weights2)

        #print (res2)

        #weightedSums2 = [np.sum([x * n, y * n, r * n, n], axis=0) for x, y, r, n in circles]
        #finalCircle = [el/weightedSums2[3] for el in weightedSums2]

        #print (finalCircle)
        return finalCircle[0], finalCircle[1], finalCircle[2]

    else:
        return None, None, None

def outliers_elimination_with_bins(img_shape, circles, bins):
    # img_shape: rows and columns
    # circles: list of tuple (center x, center y, radius, number of pixel)
    # bins: tuple of number of bins ((axisX, axisY), radius)

    # input numero bin o grandezza bin?
    # tarare dimensione blobs
    # tarare dimensione bins
    # interpolare blob vincenti

    votes_bins = np.zeros((bins[0][0], bins[0][1], bins[1]))
    circle_bins = [[[[] for _ in range(bins[1])] for _ in range(bins[0][1])] for _ in range(bins[0][0])]

    bin_shape_rows = img_shape[0] // bins[0][0]
    bin_shape_cols = img_shape[1] // bins[0][1]
    bin_shape_r = img_shape[0] // 2 // bins[1]

    for circle in circles:
        row_bin = np.round(circle[0]).astype("int") // bin_shape_rows
        col_bin = np.round(circle[1]).astype("int") // bin_shape_cols
        r_bin = np.round(circle[2]).astype("int") // bin_shape_r

        votes_bins[row_bin][col_bin][r_bin] += circle[3]
        circle_bins[row_bin][col_bin][r_bin].append(circle)

    maximum = np.unravel_index(np.argmax(votes_bins, axis=None), votes_bins.shape)
    remaining_circles = circle_bins[maximum[0]][maximum[1]][maximum[2]]

    print('CIRCLES:' + str(len(circles)))
    print('REMAINING CIRCLES:' + str(len(remaining_circles)))

    # weighted = [[x * n, y * n, r * n, n] for x, y, r, n in remaining_circles]
    # sums = [sum(a) for a in zip(*weighted)]
    # mean_circle = [el/sums[3] for el in sums]

    blob_x = [x for circle in remaining_circles for x in circle[4][0]]
    blob_y = [y for circle in remaining_circles for y in circle[4][1]]

    x, y, r = circledetection.leastSquaresCircleFitCached(blob_x, blob_y)

    # return mean_circle[0], mean_circle[1], mean_circle[2]
    return x, y, r

#@profile
def test():
    blobs = labelling.bestLabelling('d_16.bmp')

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
        x, y, r = circledetection.leastSquaresCircleFitCached(blob[0], blob[1])
        #print (x)
        #print (y)
        #print (r)

        if not (math.isnan(x) or math.isnan(y) or math.isnan(r)):
            #cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
            #cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
            #cv2.imshow('circles', img)
            #cv2.waitKey()
            circles.append((x, y, r, len(blob[0])))

    print (len(circles))
    x, y, r = outliersElimination(circles, (50, 30))
    #cv2.circle(img, (int(y), int(x)), int(r), (0, 255, 0), 1)
    #cv2.circle(img, (int(y), int(x)), 2, (0, 0, 255), 3)
    #cv2.imshow('circles', img)
    #cv2.waitKey()

if __name__ == '__main__':
    #circles = [(281.3, 390.7, 204.7, 1890), (276.5, 387.4, 201.6, 1993), (285.9, 382.5, 200.4, 2123), (281.9, 375.5, 201.4, 2004), (1234.9, 1435.3, 54.7, 127)]
    #thresholds = (50, 50)

    #for i in range (0, 1):
    #    a = outliersElimination(circles, thresholds)
    #    print (a)

    for i in range(0, 1):
        test()
    #    print (i)