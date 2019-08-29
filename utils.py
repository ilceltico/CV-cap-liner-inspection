import numpy as np
import cv2

HARALICK_THRESHOLD = 200

def binarize(img):
    """
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
    """

    retVal, labels = cv2.connectedComponentsWithAlgorithm(edges, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
  
    nonzero = np.nonzero(labels)
    #print(nonzero)
    #print(retVal)
    retVal -= 1

    blobsX = [[] for i in range(retVal)]
    blobsY = [[] for i in range(retVal)]

    for i in range(0, len(nonzero[0])):
        x = nonzero[1][i]
        y = nonzero[0][i]

        p = labels[x][y]

        blobsX[p - 1].append(x)
        blobsY[p - 1].append(y)

    result = []
    
    for i in range(retVal):
        result.append((blobsX[i], blobsY[i]))

    return result


def circularmask(image_height, image_width, center=None, radius=None):
    """
    """

    if center is None: # use the middle of the image
        center = [int(image_width/2), int(image_height/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], image_width-center[0], image_height-center[1])

    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_missing_liner_threshold():
    """
    """

	#First element is the average of perfect caps, second element of missing liners
    average = [0,0]
    i = 0
    for fileStart in ['g', 'd_31']:
        prefixed = [filename for filename in os.listdir('./caps') if filename.startswith(fileStart)]
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


