import utils
import circledetection
import os
import math
import cv2
import numpy as np
import loadconfiguration as config
import traceback

def outer_circle_detection(img):
    r"""
    Find the outer circle of the cap.

    Parameters:
        img: the image.

    Returns:
        Center x coordinate, center y coordinate and radius.
    """
    #Temporary_comment: expects the binarized image

    # Hough Transform
    if config.OUTER_METHOD == 'hough':

        # Compute the HT and average out the best circles, as many as specified in the config
        x, y, r = circledetection.find_circles_hough(img, 1, 1, 200, 5, 0, 
                0, config.OUTER_HOUGH_NUMBER_AVG)

        if config.OUTER_RADIUS_COMPUTATION == 'border_distance':
            # Take the center as an average of the best circles, and compute the radius subsequently
            # This is because, according to the official OpenCV doc, the radius computation in its function is not precise.
            # Compute the radius as the mean distance between points and the previously found center
            edges = cv2.Canny(img, 100, 200, L2gradient=True)
            pixels_y, pixels_x = np.nonzero(edges)
            
            r = sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)

    
    # Least Squares Linear Regression
    else:
        edges = cv2.Canny(img, 100, 200, apertureSize=3, L2gradient=True)

        x, y, r = circledetection.find_circle_ols(edges, 0, "mean", 
            config.OUTER_LEAST_SQUARES_CIRCLE_GENERATION, oe_thresholds=(20,20))

    return x, y, r


def inner_circle_detection(img, outer_xc, outer_yc, outer_r):
    r"""
    Find the innet circle of the cap.

    Parameters:
        img: the image.
        outer_xc: outer circle center x coordinate.
        outer_yc: outer circle center y coordinate.
        outer_yc: outer circle radius.

    Returns:
        Center x coordinate, center y coordinate and radius.
    """
    #Temporary_comment: expects the stretched image

    # Hough Transform
    if config.INNER_METHOD == 'hough':    

        # Precise Canny method:
        #   This method computes a precise Canny edge detection with L2 gradients, because the usual
        #   detection inside HoughCircles uses L1 gradients, which is quite imprecise for our purposes.
        if config.INNER_CANNY_PRECISION == 'precise':
            gaussian = cv2.GaussianBlur(img, (7,7), 2, 2)
            edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

            # Delete edge points that belong to the outer circle
            mask = utils.circular_mask(gaussian.shape[0], gaussian.shape[1], (outer_xc, outer_yc), 0.95*outer_r)
            edges[~mask] = 0

            # Compute the HT and average out the best circles, as many as specified in the config
            x, y, r = circledetection.find_circles_hough(edges, 1, 1, 100, 10, 0, 
                np.round(0.98*outer_r).astype("int"), config.INNER_HOUGH_NUMBER_AVG)

        # Normal method
        else:
            gaussian = cv2.GaussianBlur(img, (9,9), 2, 2)

            # Compute the HT and average out the best circles, as many as specified in the config
            x, y, r, = circledetection.find_circles_hough(gaussian, 1, 1, 100, 10, 0, 
                np.round(0.9*outer_r).astype("int"), config.INNER_HOUGH_NUMBER_AVG)


    # Least Squares Linear Regression
    else:
        gaussian = cv2.GaussianBlur(img, (7,7), 2, 2)
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

        mask = utils.circular_mask(gaussian.shape[0], gaussian.shape[1], (outer_xc, outer_yc), 0.95*outer_r)
        edges[~mask] = 0

        if config.INNER_LEAST_SQUARES_SPLIT == True:
            min_blob_dim = config.INNER_LEAST_SQUARES_MIN_BLOB_DIM
        else:
            min_blob_dim = 0

        x, y, r = circledetection.find_circle_ols(edges, min_blob_dim, config.INNER_LEAST_SQUARES_OUTLIERS_TYPE, 
            config.INNER_LEAST_SQUARES_CIRCLE_GENERATION, oe_thresholds=(20,20), oe_bins_factor=8)

    return x, y, r
    

#def liner_defects_detection(img):
def liner_defects_detection(stretched, liner_xc, liner_yc, liner_r):
    r"""
    Detect the defects inside the cap liner, if presents.

    Parameters:
        stretched the stretched image.
        liner_xc: cap liner center x coordinate.
        liner_yc: cap liner center y coordinate.
        liner_r: cap liner radius.

    Returns:
        Two values: a boolean and a list. If the first value is True the list will contains the rectangles contours inscribing the defects (as a list of points), otherwise the list is empty.
    """

    gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)

    edges = cv2.Canny(gaussian, 20, 110, apertureSize=3, L2gradient=True)
    mask = utils.circular_mask(stretched.shape[0], stretched.shape[1], (liner_xc, liner_yc), 0.95*liner_r)
    edges[~mask] = 0

    has_defects = False
    blobs = utils.get_blobs(edges)

    liner = np.zeros((stretched.shape[0],stretched.shape[1]), dtype=np.uint8)
    cv2.circle(liner, (np.round(liner_yc).astype("int"), np.round(liner_xc).astype("int")), np.round(0.95*liner_r).astype("int"), (255, 255, 255), 2)
    nonzero = np.nonzero(liner)
    liner = list(zip(nonzero[0],nonzero[1]))

    rectangles = []

    for blob in blobs:
        common = list(set(liner).intersection(list(zip(blob[0],blob[1]))))
        max_distance = 0
        for pixel in common:
            for pixel2 in common:
                distance = math.sqrt((pixel[0]-pixel2[0])**2 + (pixel[1]-pixel2[1])**2)
                if distance > max_distance:
                    max_distance = distance

        if len(common) >= 2 and max_distance > liner_r/10:
            has_defects = True
            rect = cv2.minAreaRect(np.array(list(zip(blob[0], blob[1]))))
            rect_dim = rect[1]
            #Increase the smaller dimension of the rect, to make it more visible.
            if rect_dim[0] < rect_dim[1]:
                rect_dim = (rect_dim[0]*2, rect_dim[1]*1)
            else:
                rect_dim = (rect_dim[0]*1, rect_dim[1]*2)
            rect = (rect[0], rect_dim, rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rectangles.append(box)

    return has_defects, rectangles

def main():
    r"""
    The main program.
    """
    
    config.parse_json()

    print("Configuration correctly loaded.")

    missing_liner_threshold = utils.get_missing_liner_threshold()
    print("Missing liner threshold: " + "%.2f" % round(missing_liner_threshold,2))
    
    for file in os.listdir('./caps'):
        # if (file != "g_06.bmp" and file != "g_01.bmp"):
        #     continue
        print('--------------------------------------------------')
        print(file)

        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)

        # Image binarization
        binary = utils.binarize(img)
        #cv2.imshow('binary', binary); cv2.waitKey(0); cv2.destroyAllWindows()

        # Cap mask creation
        mask = binary.copy().astype(bool)

        # Test if the cap is a circle
        circularity = utils.haralick_circularity(binary)
        if circularity >= config.HARALICK_THRESHOLD:
            print("The cap in " + file + " is a circle, Haralick's Circularity = " + "%.2f" % round(circularity,2))
        else:
            print("The cap in " + file + " is NOT a circle, Haralick's Circularity = " + "%.2f" % round(circularity,2))

        # TASK1
        print('TASK1')

        # Determine outer circle
        outer_xc, outer_yc, outer_r = outer_circle_detection(binary)

        if not (outer_xc is None or outer_yc is None or outer_r is None):
            print('Position of the center of the cap: (' + "%.2f" % round(outer_xc,2) + ', ' + "%.2f" % round(outer_yc,2) + ')')
            print('Diameter of the cap: ' + "%.2f" % round(2*outer_r,2))

            # Show the outer circle
            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(outer_xc).astype('int'), np.round(outer_yc).astype('int')), np.round(outer_r).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(outer_xc).astype('int'), np.round(outer_yc).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' outer circle (cap)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Is the liner missing?
            avg = np.mean(img[mask])
            if avg > missing_liner_threshold:
                print(file + ' has NO liner')
                continue
            else:
                print(file + ' has the liner')

        # TASK2
        print('TASK2')

        # Linear stretching (with mask)
        stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
        stretched[~mask] = 0

        # Determine inner circle
        liner_xc, liner_yc, liner_r = inner_circle_detection(stretched, outer_xc, outer_yc, outer_r)

        if not (liner_xc is None or liner_yc is None or liner_r is None):
            print('Position of the center of the liner: (' + "%.2f" % round(liner_xc,2) + ', ' + "%.2f" % round(liner_yc,2) + ')')
            print('Diameter of the liner: ' + "%.2f" % round(2*liner_r,2))

            # Show the inner circle
            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(liner_xc).astype('int'), np.round(liner_yc).astype('int')), np.round(liner_r).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(liner_xc).astype('int'), np.round(liner_yc).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' inner circle (liner)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # DEFECT DETECTION
            has_defects, rectangles = liner_defects_detection(stretched, liner_xc, liner_yc, liner_r)

            print(rectangles)

            if not has_defects :
                print(file + ' has no defects: the liner is complete')
            else:
                print(file + ' has defects: the liner is incomplete')
                # Show the straight edge of the liner
                coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for rectangle in rectangles:
                    cv2.drawContours(coloured_image, [rectangle], 0, (0,0,255), 1)
                cv2.imshow('caps/' + file + ' detected defects', coloured_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()