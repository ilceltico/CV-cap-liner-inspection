import numpy as np
import cv2
import json
import os

HARALICK_THRESHOLD = 200

def binarize(img):
    """
    Divides the image in two regions, using Otsu's algorithm.

    Parameters:
        img: the image.

    Returns:
        A binarized image.
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


def is_circle(binary):
    """
    Determines if a binarized image has a circular form or not.
    
    Parameters:
        binary: the binarized image.

    Returns:
        True if the form is circular enough, False otherwise.
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
    print('Circularity: ' + str(haralick_circularity))

    # Test Haralick's circularity
    if haralick_circularity >= HARALICK_THRESHOLD :
        #print('the cap is a circle')
        return True
    else:
        return False


def get_blobs(edges):
    """
    Finds connected blobs from a contour/edge image.

    Parameters:
        edges: the image with edges/contours.

    Returns:
        A list of tuples. Each tuple is a blob and is made of two arrays: the first is for x coordinates, the second for y coordinates.
    """

    ret_val, labels = cv2.connectedComponentsWithAlgorithm(edges, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
  
    nonzero = np.nonzero(labels)
    #print(nonzero)
    #print(retVal)

    # subtraction as ret_val contains also background label
    ret_val -= 1

    blobs_x = [[] for i in range(ret_val)]
    blobs_y = [[] for i in range(ret_val)]

    for i in range(0, len(nonzero[0])):
        x = nonzero[0][i]
        y = nonzero[1][i]

        p = labels[x][y]

        blobs_x[p - 1].append(x)
        blobs_y[p - 1].append(y)

    #result = []
    
    #for i in range(ret_val):
    #    result.append((blobs_x[i], blobs_y[i]))

    result = [(x,y) for x, y in zip(blobs_x, blobs_y)]

    return result


def circular_mask(img_height, img_width, center=None, radius=None):
    """
    Produces a binary circular mask with the specified parameters.
    
    Parameters:
        image_height: int.
        image_width: int.
        center: couple of coordinates (x,y).
        radius: positive number.

    Returns:
        A binary image with True inside the specified circle, False outside.
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
    """
    Computes the threshold to determine if a liner is missing or not by analyzing existing supervised examples.

    Returns:
        The found threshold.
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

def parse_json():
    with open('config.json') as file:
        config = json.load(file)
        circle_detection_param = config['circle_detection']
        outer = circle_detection_param['outer']
        outer_method = outer['method']
        if not (outer_method in ["hough", "least_squares"]):
            print("Configuration error. See README.md to configure properly the software.")
            return None

        outer_parameters = outer['parameters']
        if outer_method == "least_squares" and not (outer_parameters['circle_generation'] in ["mean", "interpolation"]):
            print("Configuration error. See README.md to configure properly the software.")
            return None

        inner = circle_detection_param['inner']
        inner_method = inner['method']
        if not (inner_method in ["hough", "least_squares"]):
            print("Configuration error. See README.md to configure properly the software.")
            return None

        inner_parameters = inner['parameters'][inner_method]
        if inner_method == 'hough':
            if not (inner_parameters['image_to_hough'] in ["edges", "gaussian"]) or not (isinstance(inner_parameters['number_of_circle_average'], int)) or inner_parameters['number_of_circle_average'] > 2 or inner_parameters['number_of_circle_average'] < 1:
                print("Configuration error. See README.md to configure properly the software.")
                return None
        else:
            if not (isinstance(inner_parameters['split_blobs'], bool)) or not (inner_parameters['outliers_elimination_type'] in ["mean", "bin"]) or not (inner_parameters['circle_generation'] in ["mean", "interpolation", "interpolation_cook"]) or (inner_parameters['split_blobs'] == True and inner_parameters['outliers_elimination_type'] == "mean"):
                print("Configuration error. See README.md to configure properly the software.")
                return None
         
        print("Configuration: all ok")
        return circle_detection_param