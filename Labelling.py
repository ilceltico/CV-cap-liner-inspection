import cv2
import numpy as np
import os

#file = 'test.bmp'
file = 'd_16.bmp'

img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 180, 160, apertureSize=3, L2gradient=True)

#edges = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

rows, cols = edges.shape
firstX = 0
firstY = 0

edgesList = edges.tolist()

for row in range(0, rows):
    for col in range(0, cols):
        if edgesList[row][col] == 255:
            firstX = row
            firstY = col
            break
    else:
        continue
    break

queue = {(firstX, firstY)}
result = []

while queue:
    x, y = queue.pop()
    edgesList[x][y] = 0

    for row in range(x - 1, x + 2):
        for col in range(y - 1, y + 2):
            if edgesList[row][col] == 255:
                queue.add((row, col))

    #[queue.append((x, y)) for sublist in edgesList[x-1:x+2] for pixel in sublist[col-1:col+2] if pixel == 255]

    result.append((x, y))