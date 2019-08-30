import utils
import circledetection
import os
import math
import cv2
import numpy as np

def circle_detection(edges):


def outer_circle_detection(img):
    # bynary mask + canny + labelling + least square + outliers (media o interpolazione finale?) (tests.py)
    # binary mask + hough + 3 circles and mean of distance from canny (tests_hough_version.py)

    x = None
    y = None
    r = None

    binary = binarization.binarize(img)
    mask = binary.copy().astype(bool)

    # ----------
    # if least square
    # ----------

    edges = cv2.Canny(binary, 100, 200, apertureSize=3, L2gradient=True)
    blobs = utils.get_blobs(edges)

    circles = []

    for blob in blobs:
        x, y, r = circledetection.least_squares_circle_fit(blob[0], blob[1])
        if not (math.isnan(x) or math.isnan(y) or math.isnan(r)):
            circles.append((x, y, r, len(blob[0]), blob))
    
    remaining_circles = circledetection.outliers_elimination_blobs(circles, (20, 20))

    blob_x = [x for circle in circles_remaining for x in circle[4][0]]
    blob_y = [y for circle in circles_remaining for y in circle[4][1]]

    x, y, r = circledetection.least_squares_circle_fit(blob_x, blob_y)
    
    # ----------
    # if hough
    # ----------

    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=5, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    
    #   take only the center from the Hough result and compute the radius
    x = sum([circles[0][i][1] for i in range(3)]) / 3
    y = sum([circles[0][i][0] for i in range(3)]) / 3

    # compute the radius as the mean distance between points and the center (found with Hough)
    edges = cv2.Canny(binary, 100, 200, L2gradient=True)
    pixels_x, pixels_y = np.nonzero(edges)

    r = sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)

    return x, y, r

def inner_circle_detection(img):
    # mask + stretching + mask + gaussian + canny + labelling + least square + outliers (media o interpolazione finale?) (tests.py)
    # HOUGH EDGES
    # mask + stretching + mask + gaussian + canny + labelling + blobs split + least square + bin outliers (test_with_bin_test.py)
    # mask + stretching + mask + gaussian + canny + labelling + least square + outliers with remaining circles + iteration with cook least square (test2.py)
    # HOUGH GAUSSIAN AND TAKE FIRST
    # hough + 3 circles and mean of distance from canny    

def missing_liner_detection(img, threshold):
    # missing_liner_threshold = utils.get_missing_liner_threshold()

    binary = utils.binarize(img)
    mask = binary.copy().astype(bool)
    avg = np.mean(img[mask])

    if avg > threshold:
        return True
    else:
        return False

def liner_defects_detection(img):
    # TO JUST SAY YES: binary mask + stretching + mask + gaussian + canny + mask + hough lines
    # mask + stretching + mask + gaussian + canny + mask + labelling + intersection and distance
    pass

def execute():
    missing_liner_threshold = utils.get_missing_liner_threshold()
    print('Missing liner threshold: ' + str(missing_liner_threshold))

    for file in os.listdir('./caps'):
        print('--------------------------------------------------')
        print(file)

        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        # binary = utils.binarize(img)
        #cv2.imshow('binary', binary)

        # test if the cap is a circle
        if not utils.is_circle(binary):
            print('The cap in ' + file + ' is NOT a circle')
            continue
        else:
            print('The cap in ' + file + ' is a circle')

        # TASK1
        print('TASK1')

        x_cap, y_cap, r_cap = outer_circle_detection(img)

        if not (x_cap is None or y_cap is None or r_cap is None):
            print('Position of the center of the cap: (' + str(x_cap) + ', ' + str(y_cap) + ')')
            print('Diameter of the cap: ' + str(2 * r_cap))

            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(y_cap).astype('int'), np.round(x_cap).astype('int')), np.round(r_cap).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(y_cap).astype('int'), np.round(x_cap).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' outer circle (cap)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print('Is the liner missing?')

            missing_liner = missing_liner_detection(img, missing_liner_threshold)

            if missing_liner:
                print('caps/' + file + ' has NO liner')
                continue
            else:
                print('caps/' + file + ' has liner')

            # mask = binary.copy().astype(bool)
            # avg = np.mean(img[mask])
            # print('caps/' + file + ' pixels average: ' + str(avg))

            # if avg > missing_liner_threshold:
            #     print('caps/' + file + ' has NO liner')
            #     continue
            # else:
            #     print('caps/' + file + ' has liner')

        # TASK2
        print('TASK2')

        x_liner, y_liner, r_liner = inner_circle_detection(img)

        if not (x_liner is None or y_liner is None or r_liner is None):
            print('Position of the center of the liner: (' + str(x_liner) + ', ' + str(y_liner) + ')')
            print('Diameter of the liner: ' + str(2 * r_liner))

            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(y_liner).astype('int'), np.round(x_liner).astype('int')), np.round(r_liner).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(y_liner).astype('int'), np.round(x_liner).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' inner circle (liner)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # DEFECT DETECTION
            print('Is the liner incomplete?')

            has_defects, rectangle = liner_defects_detection(img)

            if not has_defects :
                print('caps/' + file + ' has NO defects')
            else:
                print('caps/' + file + ' has defects')
                coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(coloured_image, [rectangle], 0, (0,0,255), 1)
                cv2.imshow('caps/' + file + ' detected defects', coloured_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    execute()