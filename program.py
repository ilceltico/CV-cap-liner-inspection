import utils
import circledetection
import os
import math
import cv2
import numpy as np
import readConfiguration as config


HARALICK_THRESHOLD = 200

def circle_detection(img, r_cap=None):
    """
    Perform circle detection.

    Parameters:
        img: input image.
        r_cap: if specified the inner circle will be found (if present) otherwise the outer circle will be detected.

    Returns:
        Center x coordinate, center y coordinate and radius
    """

    if r_cap == None:
        outer_circle_detection(img)
    else:
        inner_circle_detection(img, r_cap)
    
#def outer_circle_detection(img):
def outer_circle_detection(binary):

    x = None
    y = None
    r = None

    #binary = binarization.binarize(img) #to delete

    # Hough Transform
    if config.OUTER_METHOD == 'hough':
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=5, minRadius=0, maxRadius=0)
        #circles = np.uint16(np.around(circles)) #to delete
        
        # Take the center as an average of the 3 best circles, and compute the radius subsequently
        # This is because, according to the official OpenCV doc, the radius computation in its function is not precise.
        number_of_circles = 3
        if len(circles[0]) <= 0:
            print("No circles found")
            return x, y, r
        if len(circles[0]) < number_of_circles:
            number_of_circles = len(circles[0])
            print("WARNING: less than the specified amount of circles was found, using " + str(number_of_circles) + " circles only.")

        x = sum([circles[0][i][0] for i in range(number_of_circles)]) / number_of_circles
        y = sum([circles[0][i][1] for i in range(number_of_circles)]) / number_of_circles

        # Compute the radius as the mean distance between points and the previously found center
        edges = cv2.Canny(binary, 100, 200, L2gradient=True)
        pixels_y, pixels_x = np.nonzero(edges)

        r = sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
    
    # Least Squares Linear Regression
    else:
        edges = cv2.Canny(binary, 100, 200, apertureSize=3, L2gradient=True)
        blobs = utils.get_blobs(edges)

        circles = []

        # Weighted mean method
        if config.OUTER_LEAST_SQUARES_CIRCLE_GENERATION == 'mean':  
            # Find a circle fit for each blob     
            for blob in blobs:
                x_temp, y_temp, r_temp = circledetection.least_squares_circle_fit(blob[0], blob[1])
                if not (math.isnan(x_temp) or math.isnan(y_temp) or math.isnan(r_temp)):
                    circles.append((x_temp, y_temp, r_temp, len(blob[0])))
            
            # Eliminate circles that are too far away from the weighted mean
            remaining_circles = circledetection.outliers_elimination(circles, (20, 20))

            # Re-compute a weighted mean circle
            weighted = [[circle[0] * circle[3], circle[1] * circle[3], circle[2] * circle[3], circle[3]] for circle in remaining_circles]
            sums = [sum(a) for a in zip(*weighted)]
            x, y, r, _ = [el/sums[3] for el in sums]

        # Full interpolation method
        else:
            # Find a circle fit for each blob
            for blob in blobs:
                x_temp, y_temp, r_temp = circledetection.least_squares_circle_fit(blob[0], blob[1])
                if not (math.isnan(x_temp) or math.isnan(y_temp) or math.isnan(r_temp)):
                    circles.append((x_temp, y_temp, r_temp, len(blob[0]), blob))
            
            # Eliminate circles that are too far away from the weighted mean
            remaining_circles = circledetection.outliers_elimination(circles, (20, 20))
            
            # Merge the remaining blobs
            blob_x = [x for circle in remaining_circles for x in circle[4][0]]
            blob_y = [y for circle in remaining_circles for y in circle[4][1]]

            # Fit again
            x, y, r = circledetection.least_squares_circle_fit(blob_x, blob_y)

    return x, y, r

#def inner_circle_detection(img, r_cap):
def inner_circle_detection(stretched, r_cap):

    x = None
    y = None
    r = None

    #binary = utils.binarize(img)
    #mask = binary.copy().astype(bool)
    #stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
    #stretched[~mask] = 0
    #to delete

    # Hough Transform
    if config.INNER_METHOD == 'hough':    

        # Precise Canny method:
        #   This method computes a precise Canny edge detection with L2 gradients, because the usual
        #   detection inside HoughCircles uses L1 gradients, which is quite imprecise.
        if config.INNER_HOUGH_IMAGE == 'edges':
            gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
            edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=0, maxRadius=np.round(0.98*r_cap).astype("int"))
            #circles = np.uint16(np.around(circles)) #to delete

            # Take the center as an average of the best circles
            number_of_circles = config.INNER_HOUGH_NUMBER_AVG
            if len(circles[0]) <= 0:
                print("No circles found!")
                return x, y, r
            if len(circles[0]) < number_of_circles:
                number_of_circles = len(circles[0])
                print("WARNING: less than the specified amount of circles was found, using " + str(number_of_circles) + " circles only.")

            x = sum(circles[0][i][0] for i in range(number_of_circles)) / number_of_circles # try np.mean()
            y = sum(circles[0][i][1] for i in range(number_of_circles)) / number_of_circles
            r = sum(circles[0][i][2] for i in range(number_of_circles)) / number_of_circles

        # Gaussian-only method
        else:
            gaussian = cv2.GaussianBlur(stretched, (9,9), 2, 2)

            circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=0, maxRadius=np.round(0.9*r_cap).astype("int"))
            #circles = np.uint16(np.around(circles)) #to delete

            # Return the best circle found
            x = circles[0][0][0]
            y = circles[0][0][1]
            r = circles[0][0][2]

    # Least Squares Linear Regression
    else:
        gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
        edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)

        blobs = utils.get_blobs(edges)

        circles = []

        # not split
        if not config.INNER_LEAST_SQUARES_SPLIT:
            for blob in blobs:
                x_temp, y_temp, r_temp = circledetection.least_squares_circle_fit(blob[0], blob[1])
                if not (math.isnan(x_temp) or math.isnan(y_temp) or math.isnan(r_temp)):
                    if r_temp < 0.99*r_cap:
                        circles.append((x_temp, y_temp, r_temp, len(blob[0]), blob))

        # split
        else:
            # coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            split_number = 8

            for blob in blobs:
                blob_x = blob[0]
                blob_y = blob[1]
                length = int(len(blob_x) / split_number) + 1
                for i in range(split_number):
                #     img_copy = coloured_image.copy()
                #     color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
                #     for j in range(0, length):
                #         if j+i*length < len(blobX):
                #             img_copy[blob_x[j+i*length]][blob_y[j+i*length]] = color

                    if length - 1 + i * length >= len(blob_x):
                        max_index = len(blob_x)-1
                    else:
                        max_index = length - 1 + i * length
                    x_temp, y_temp, r_temp = circledetection.least_squares_circle_fit(blob_x[i * length:max_index + 1], blob_y[i * length:max_index + 1])
                    if not (math.isnan(x_temp) or math.isnan(y_temp) or math.isnan(r_temp)):
                        if r_temp < 0.99*r_cap:
                            circles.append((x_temp, y_temp, r_temp, len(blob_x[i * length:max_index]), blob))

        # mean for outliers elimination
        if config.INNER_LEAST_SQUARES_OUTLIERS_TYPE == 'mean':
            remaining_circles = circledetection.outliers_elimination(circles, (20,20))
        
        # bin
        else:
            #remaining_circles = circledetection.outliers_elimination_with_bins(img.shape[0], img.shape[1], circles, (72, 85, 36))  
            remaining_circles = circledetection.outliers_elimination_with_bins(stretched.shape[0], stretched.shape[1], circles, (72, 85, 36)) 

        # # ELSE no outliers elimination

        #     # outliers MERGED
        #     blob_x = [x for circle in circles for x in circle[4][0]]
        #     blob_y = [y for circle in circles for y in circle[4][1]]

        #         # interpolation with LEAST SQUARE
        #         x, y, r = circledetection.least_squares_circle_fit(blob_x, blob_y)

        #         # interpolation with COOK
        #         x, y, r, cook_d = # iteration with cook elimination are needed


        # mean for circle generation
        if config.INNER_LEAST_SQUARES_CIRCLE_GENERATION == 'mean':
            weighted = [[circle[0] * circle[3], circle[1] * circle[3], circle[2] * circle[3], circle[3]] for circle in remaining_circles]
            sums = [sum(a) for a in zip(*weighted)]
            x, y, r, _ = [el/sums[3] for el in sums]

        else:
            blob_x = [x for circle in remaining_circles for x in circle[4][0]]
            blob_y = [y for circle in remaining_circles for y in circle[4][1]]

            # interpolation
            if config.INNER_LEAST_SQUARES_CIRCLE_GENERATION == 'interpolation':
                x, y, r = circledetection.least_squares_circle_fit(blob_x, blob_y)
            
            # cook
            # else:
            #     x, y, r, cook_d = # iteration with cook elimination are needed

    return x, y, r
    

#def liner_defects_detection(img):
def liner_defects_detection(stretched, x_liner, y_liner, r_liner):
    """
    """

    # TO JUST SAY YES: binary mask + stretching + mask + gaussian + canny + mask + hough lines
    # mask + stretching + mask + gaussian + canny + mask + labelling + intersection and distance

    #binary = binarization.binarize(img)
    #mask = binary.copy().astype(bool)

    #stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
    #stretched[~mask] = 0

    gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)

    edges = cv2.Canny(gaussian, 20, 110, apertureSize=3, L2gradient=True)
    mask = utils.circular_mask(stretched.shape[0], stretched.shape[1], (x_liner, y_liner), 0.95*r_liner)
    edges[~mask] = 0

    has_defects = False
    blobs = utils.get_blobs(edges)

    #liner = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    liner = np.zeros((stretched.shape[0],stretched.shape[1]), dtype=np.uint8)
    cv2.circle(liner, (np.round(y_liner).astype("int"), np.round(x_liner).astype("int")), np.round(0.95*r_liner).astype("int"), (255, 255, 255), 2)
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

        if len(common) >= 2 and max_distance > r_liner/10:
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

def execute():
    try:
        config.parse_json()
    except:
        raise
    print("Configuration correctly loaded.")

    missing_liner_threshold = utils.get_missing_liner_threshold()
    print('Missing liner threshold: ' + str(missing_liner_threshold))

    for file in os.listdir('./caps'):
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
        if circularity >= HARALICK_THRESHOLD :
            print("The cap in " + file + " is a circle, Haralick's Circularity = " + "%.2f" % round(circularity,2))
        else:
            print("The cap in " + file + " is NOT a circle, Haralick's Circularity = " + "%.2f" % round(circularity,2))

        # TASK1
        print('TASK1')

        # Determine outer circle
        x_cap, y_cap, r_cap = outer_circle_detection(binary)

        if not (math.isnan(x_cap) or math.isnan(y_cap) or math.isnan(r_cap)):
            print('Position of the center of the cap: (' + "%.2f" % round(x_cap,2) + ', ' + "%.2f" % round(y_cap,2) + ')')
            print('Diameter of the cap: ' + "%.2f" % round(2*r_cap,2))

            # Show the outer circle
            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(x_cap).astype('int'), np.round(y_cap).astype('int')), np.round(r_cap).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(x_cap).astype('int'), np.round(y_cap).astype('int')), 2, (0, 0, 255), 3)
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

        #x_liner, y_liner, r_liner = inner_circle_detection(img, r_cap=r_cap) #to delete

        # Linear stretching (with mask)
        stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
        stretched[~mask] = 0

        # Determine inner circle
        x_liner, y_liner, r_liner = inner_circle_detection(stretched, r_cap=r_cap)

        if not (math.isnan(x_liner) or math.isnan(y_liner) or math.isnan(r_liner)):
            print('Position of the center of the liner: (' + "%.2f" % round(2*x_liner,2) + ', ' + "%.2f" % round(2*y_liner,2) + ')')
            print('Diameter of the liner: ' + "%.2f" % round(2*r_liner,2))

            # Show the inner circle
            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(x_liner).astype('int'), np.round(y_liner).astype('int')), np.round(r_liner).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(x_liner).astype('int'), np.round(y_liner).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' inner circle (liner)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # DEFECT DETECTION
            #has_defects, rectangles = liner_defects_detection(img) #to delete
            has_defects, rectangles = liner_defects_detection(stretched, x_liner, y_liner, r_liner)

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
        execute()
    except:
        pass