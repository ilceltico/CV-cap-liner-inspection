import numpy as np
import cv2
import json
import os

def binarize(img):
    r"""
    Divides the image in two regions using Otsu's algorithm.

    Parameters
    ----------
        img: 2d array
            the image.

    Returns
    -------
        closing: 2d array
            the binarized image.
    """

    # Show bimodal histogram
    # hist = cv2.calcHist([img], [0], None, [256], [0,256])
    # plt.plot(hist)
    # plt.show()

    #Otsu's Thresholding
    ret, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print ('threshold found: ' + str(ret)) #print the threshold found with otsu (ret)
        
    #Morphological Closing operation
    kernel = np.ones((7,7), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))    #using an elliptical kernel shape (obtain the same result)
    closing = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('closing', closing)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return closing


def haralick_circularity(binary):
    r"""
    Returns the Haralick's Circularity of the specified binarized image.
    
    Parameters
    ----------
        binary: 2d array
            the binarized image.

    Returns
    -------
        haralick_circularity: float
            haralick circularity value.
    """

    # Contours are simply a curve joining all the points (along the boundary), having same color or intensity.
    #first argument: image
    #second argument: contour retrieval mode.
    #third argument: contour approximation method. Two possibilities: 
    #   cv2.CHAIN_APPROX_NONE: all the boundary points are stored.
    #   cv2.CHAIN_APPROX_SIMPLE: removes all redundant points and compresses the contour, saving memory. E.g. for a line store only the two end points.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #Should find 4-connected perimeters

    cnt = contours[0]
    # print(cnt)
    # vis = np.zeros((len(img),len(img[0])), np.uint8)
    # cv2.drawContours( vis, contours, (-1, 2)[3 <= 0], (128,255,255),
    #         1, cv2.LINE_AA, hierarchy, abs(3) )
    # cv2.imshow('contours', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    #Haralick's Circularity
    moments = cv2.moments(cnt)
    i_b = int(moments["m10"] / moments["m00"])
    j_b = int(moments["m01"] / moments["m00"])
    # print([i_b, j_b])
    diff = cnt - np.full_like(cnt,[i_b, j_b])
    diff = (diff).reshape(len(diff),2)
    distances_from_bary = np.linalg.norm(diff, axis=1)
    average_distance = np.sum(distances_from_bary) / len(distances_from_bary)
    # print(average_distance)
    # print(moments["m00"])

    variance = np.sum(np.square(distances_from_bary - average_distance)) / len(distances_from_bary)
    # print(variance)

    haralick_circularity = average_distance / np.sqrt(variance)
    #print('Haralick's Circularity: ' + str(haralick_circularity))
    
    return haralick_circularity


def get_blobs(edges):
    r"""
    Finds connected blobs from a contour/edge image (= everything but edges is 0).

    Parameters
    ----------
        edges: 2d array
            the image with edges/contours.

    Returns
    -------
        blobs: list of tuples
            each tuple is a blob and is made of two lists: the first is for x coordinates, the second for y coordinates.
    """

    ret_val, labels = cv2.connectedComponentsWithAlgorithm(edges, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
  
    nonzero = np.nonzero(labels)
    #print(nonzero)
    #print(retVal)

    # subtraction as ret_val contains also background label
    blobs_x = [[] for i in range(ret_val - 1)]
    blobs_y = [[] for i in range(ret_val - 1)]

    for i in range(len(nonzero[0])):
        x = nonzero[1][i]
        y = nonzero[0][i]

        p = labels[y][x]

        blobs_x[p - 1].append(x)
        blobs_y[p - 1].append(y)

    blobs = [(x, y) for x, y in zip(blobs_x, blobs_y)]

    return blobs


def circular_mask(img_height, img_width, center=None, radius=None):
    r"""
    Produces a binary circular mask with the specified parameters.
    
    Parameters
    ----------
        image_height: int
            image height.
        image_width: int
            image width.
        center: tuple, optional
            x and y coordinates of circle center.
        radius: float, optional
            radius of circle.

    Returns
    -------
        mask: 2d array
            a binary image with True inside the specified circle, False outside.
    """

    if center is None: # use the middle of the image
        center = [int(img_width/2), int(img_height/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], img_width-center[0], img_height-center[1])

    Y, X = np.ogrid[:img_height, :img_width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_missing_liner_threshold():
    r"""
    Computes the threshold to determine if a liner is missing or not by analyzing existing supervised examples.

    Returns
    -------
        thresh: float
            the found threshold.
    """

	#First element is the average of perfect caps, second element of missing liners
    average = [0,0]
    i = 0
    for file_start in ['g', 'd_31']:
        prefixed = [filename for filename in os.listdir('./caps') if filename.startswith(file_start)]
        for file in prefixed:
            img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

            binary = binarize(img)
            mask = binary.copy().astype(bool)
            avg = np.mean(img[mask])

            #temp = img.copy()
            #temp[~mask] = 255
            #cv2.imshow("binarization Mask", temp)
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