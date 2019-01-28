import cv2
import numpy as np
import os

def labelling():
    #file = 'test.bmp'
    file = 'd_16.bmp'

    img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(img, 180, 160, apertureSize=3, L2gradient=True)

    #edges = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

    edgesList = edges.tolist()
    rows, cols = edges.shape
    startingX = 0
    startingY = 0

    results = []

    firstX = startingX
    firstY = startingY

    while True:
        for row in range(startingX, rows):
            for col in range(startingY, cols):
                if edgesList[row][col] == 255:
                    firstX = row
                    firstY = col
                    break
            else:
                continue
            break

        if firstX == startingX and firstY == startingY: # or i can check over results
            return results
        else:
            startingX = firstX
            startingY = firstY

        queue = {(firstX, firstY)}
        blob = []

        while queue:
            x, y = queue.pop()
            edgesList[x][y] = 0

            for row in range(x - 1, x + 2):
                for col in range(y - 1, y + 2):
                    if edgesList[row][col] == 255:
                        queue.add((row, col))

            #[queue.append((x, y)) for sublist in edgesList[x-1:x+2] for pixel in sublist[col-1:col+2] if pixel == 255]

            blob.append((x, y))

        results.append(blob)

#t1 = cv2.getTickCount()
#var = labelling()
#t2 = cv2.getTickCount()
#time = (t2 - t1)/ cv2.getTickFrequency()
#print (time)
#print (var)

#for points in var:
#    for pixel in points:
#        print (pixel)
#    print ('--------------------')

def bestLabelling(file):
    img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

    t1 =  cv2.getTickCount()

    edges = cv2.Canny(img, 180, 160, apertureSize=3, L2gradient=True)

    t2 = cv2.getTickCount()

    retVal, labels = cv2.connectedComponentsWithAlgorithm(edges, 8, cv2.CV_16U, cv2.CCL_GRANA)

    t3 = cv2.getTickCount()

    blobs = np.full(retVal - 1, None)

    for i in np.arange(retVal - 1):
        blobs[i] = np.argwhere(labels == i + 1)

    t4 = cv2.getTickCount()

    time1 = (t2 - t1)/ cv2.getTickFrequency()
    time2 = (t3 - t2)/ cv2.getTickFrequency()
    time3 = (t4 - t3)/ cv2.getTickFrequency()

    print(file)
    print ('canny: ' + str(time1))
    print ('labelling: ' + str(time2))
    print ('blob estraction: ' + str(time3))
    print ()
    print ('total: ' + str(time1 + time2 + time3))

bestLabelling('d_16.bmp')