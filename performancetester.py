import program
import circledetection
import utils
import random
import time
import os
import numpy as np
import cv2

def ols_circle_fit(x_array, y_array):
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    indip1 = 2*x_array
    indip2 = 2*y_array
    dep = x_array**2 + y_array**2
    ones = np.ones(len(x_array))

    matrix_t = np.vstack((indip1, indip2, ones))
    matrix = np.transpose(matrix_t)

    pseudoinv = np.linalg.inv(matrix_t @ matrix) @ matrix_t

    results = pseudoinv @ dep

    x_center = results[0]
    y_center = results[1]
    radius = np.sqrt(results[2] + x_center**2 + y_center**2)

    return x_center, y_center, radius


def ols_test(num_iterations):
    print('\nOLS TEST ({0} iterations)\n'.format(num_iterations))


    sets_number = 10
    sets_size = 1000
    random_range = 500

    random_points_sets = [([random.randrange(random_range) for _ in range(sets_size)], [random.randrange(random_range) for _ in range(sets_size)]) for i in range(sets_number)]

    t = time.process_time()
    for _ in range(num_iterations):
        for random_points in random_points_sets:
            x, y, r = ols_circle_fit(*random_points)
    delta = time.process_time() - t
    print('ols_circle_fit: ' + str(delta / num_iterations))

    t = time.process_time()
    for _ in range(num_iterations):
        for random_points in random_points_sets:
            x, y, r = circledetection.fast_ols_circle_fit(*random_points)
    delta = time.process_time() - t
    print('fast_ols_circle_fit: ' + str(delta / num_iterations))
    

def cook_test(num_iterations):
    print('\nCOOK TEST ({0} iterations)\n'.format(num_iterations))

    sets_number = 10
    sets_size = 1000
    random_range = 500

    random_points_sets = [([random.randrange(random_range) for _ in range(sets_size)], [random.randrange(random_range) for _ in range(sets_size)]) for i in range(sets_number)]

    t = time.process_time()
    for _ in range(num_iterations):
        for random_points in random_points_sets:
            x, y, r, cook_d = circledetection.ols_circle_cook(*random_points)
    delta = time.process_time() - t
    print('ols_circle_cook: ' + str(delta / num_iterations))

    t = time.process_time()
    for _ in range(num_iterations):
        for random_points in random_points_sets:
            x, y, r, cook_d = circledetection.fast_ols_circle_cook(*random_points)
    delta = time.process_time() - t    
    print('fast_ols_circle_cook: ' + str(delta / num_iterations))


def outer_circle_test(num_iterations):
    print('\nOUTER CIRCLE TEST ({0} iterations)\n'.format(num_iterations))

    imgs = [cv2.imread('./caps/' + file, cv2.IMREAD_GRAYSCALE) for file in os.listdir('./caps')]

    # hough avg
    t = time.process_time()
    for _ in range(num_iterations):
        for img in imgs:
            binary = utils.binarize(img)
            x, y, r = circledetection.find_circles_hough(binary, 1, 1, 200, 5, 0, 0, 3)
    delta = time.process_time() - t
    print('find_circles_hough avg: ' + str(delta / num_iterations))

    # hough avg + border distance
    t = time.process_time()
    for _ in range(num_iterations):
        for img in imgs:
            binary = utils.binarize(img)
            x, y, r = circledetection.find_circles_hough(binary, 1, 1, 200, 5, 0, 0, 3)
            edges = cv2.Canny(img, 100, 200, L2gradient=True)
            pixels_y, pixels_x = np.nonzero(edges)
            r = sum(np.sqrt((pixels_x - x)**2 + (pixels_y - y)**2)) / len(pixels_x)
    delta = time.process_time() - t
    print('find_circles_hough avg + border distance: ' + str(delta / num_iterations))

    # least squares mean
    t = time.process_time()
    for _ in range(num_iterations):
        for img in imgs:
            binary = utils.binarize(img)
            edges = cv2.Canny(binary, 100, 200, apertureSize=3, L2gradient=True)
            x, y, r = circledetection.find_circle_ols(edges, 0, "mean", "mean", oe_thresholds=(20,20))
    delta = time.process_time() - t
    print('find_circle_ols mean: ' + str(delta / num_iterations))

    # least squares regression
    t = time.process_time()
    for _ in range(num_iterations):
        for img in imgs:
            binary = utils.binarize(img)
            edges = cv2.Canny(binary, 100, 200, apertureSize=3, L2gradient=True)
            x, y, r = circledetection.find_circle_ols(edges, 0, "mean", "least_squares", oe_thresholds=(20,20))
    delta = time.process_time() - t
    print('find_circle_ols least_squares: ' + str(delta / num_iterations))    


def inner_circle_test(num_iterations):
    print('\nINNER CIRCLE TEST ({0} iterations)\n'.format(num_iterations))

    imgs = [cv2.imread('./caps/' + file, cv2.IMREAD_GRAYSCALE) for file in os.listdir('./caps')]
    edges = [cv2.Canny(img, 100, 200, apertureSize=3, L2gradient=True) for img in imgs]
    outer_circles = [circledetection.find_circle_ols(edge, 0, "mean", "least_squares", oe_thresholds=(20,20)) for edge in edges]
    masks = [utils.binarize(img).astype(bool) for img in imgs]
    imgs_outer_circles = list(zip(imgs, outer_circles, masks))

    # hough normal
    t = time.process_time()
    for _ in range(num_iterations):
        for img, outer_circle, mask in imgs_outer_circles:
            stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
            stretched[~mask] = 0
            gaussian = cv2.GaussianBlur(stretched, (9,9), 2, 2)
            x, y, r = circledetection.find_circles_hough(gaussian, 1, 1, 100, 10, 0, np.round(0.9*outer_circle[2]).astype("int"), 2)
    delta = time.process_time() - t
    print('hough normal: ' + str(delta / num_iterations))

    # hough precise
    t = time.process_time()
    for _ in range(num_iterations):
        for img, outer_circle, mask in imgs_outer_circles:
            stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
            stretched[~mask] = 0
            gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
            edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
            mask = utils.circular_mask(gaussian.shape[0], gaussian.shape[1], (outer_circle[0], outer_circle[1]), 0.95*outer_circle[2])
            edges[~mask] = 0

            if all(edges.flatten() == 0):
                continue

            x, y, r = circledetection.find_circles_hough(edges, 1, 1, 100, 10, 0, np.round(0.98*outer_circle[2]).astype("int"), 1)
    delta = time.process_time() - t
    print('hough precise: ' + str(delta / num_iterations))

    # naive mean
    t = time.process_time()
    for _ in range(num_iterations):
        for img, outer_circle, mask in imgs_outer_circles:
            stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
            stretched[~mask] = 0
            gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
            edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
            mask = utils.circular_mask(gaussian.shape[0], gaussian.shape[1], (outer_circle[0], outer_circle[1]), 0.95*outer_circle[2])
            edges[~mask] = 0
            blobs = utils.get_blobs(edges)
            circles = []

            for blob in blobs:
                x_temp, y_temp, r_temp = circledetection.fast_ols_circle_fit(blob[0], blob[1])
                if not (x_temp is None or y_temp is None or r_temp is None):
                    if x_temp >= 0 and x_temp < edges.shape[1] and y_temp >= 0 and y_temp < edges.shape[0]:
                        circles.append((x_temp, y_temp, r_temp, len(blob[0]), blob))

            remaining_circles = circledetection.outliers_elimination_mean(circles, (20,20))

            if len(remaining_circles) == 0:
                continue

            weighted = [[circle[0] * circle[3], circle[1] * circle[3], circle[2] * circle[3], circle[3]] for circle in remaining_circles]
            sums = [sum(a) for a in zip(*weighted)]
            x, y, r, _ = [el/sums[3] for el in sums]
    delta = time.process_time() - t
    print('naive mean: ' + str(delta / num_iterations))

    # least squares regression
    t = time.process_time()
    for _ in range(num_iterations):
        for img, outer_circle, mask in imgs_outer_circles:
            stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
            stretched[~mask] = 0            
            gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
            edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
            mask = utils.circular_mask(gaussian.shape[0], gaussian.shape[1], (outer_circle[0], outer_circle[1]), 0.95*outer_circle[2])
            edges[~mask] = 0
            blobs = utils.get_blobs(edges)
            circles = []

            for blob in blobs:
                blob_x = blob[0]
                blob_y = blob[1]

                if len(blob_x) < 200:
                    split_number = 1
                else :
                    split_number = len(blob_x) // 200
                length = len(blob_x) // split_number

                for i in range(split_number):
                    if i == split_number-1: #last split
                        #max index (not included)
                        max_index = len(blob_x)
                    else:
                        max_index = (i+1) * length
                    x_temp, y_temp, r_temp = circledetection.fast_ols_circle_fit(blob_x[i * length:max_index], blob_y[i * length:max_index])
                    if not (x_temp is None or y_temp is None or r_temp is None):
                        if x_temp >= 0 and x_temp < edges.shape[1] and y_temp >= 0 and y_temp < edges.shape[0]:
                            circles.append((x_temp, y_temp, r_temp, len(blob_x[i * length:max_index]), blob))

            remaining_circles = circledetection.outliers_elimination_votes(edges.shape[0], edges.shape[1], circles, 8)

            blob_x = [x for circle in remaining_circles for x in circle[4][0]]
            blob_y = [y for circle in remaining_circles for y in circle[4][1]]

            x, y, r = circledetection.fast_ols_circle_fit(blob_x, blob_y)
    delta = time.process_time() - t
    print('least squares regression: ' + str(delta / num_iterations))


    # least squares regression cook
    t = time.process_time()
    for _ in range(num_iterations):
        for img, outer_circle, mask in imgs_outer_circles:
            stretched = ((255 / (img[mask].max() - img[mask].min()))*(img.astype(np.float)-img[mask].min())).astype(np.uint8)
            stretched[~mask] = 0
            gaussian = cv2.GaussianBlur(stretched, (7,7), 2, 2)
            edges = cv2.Canny(gaussian, 45, 100, apertureSize=3, L2gradient=True)
            mask = utils.circular_mask(gaussian.shape[0], gaussian.shape[1], (outer_circle[0], outer_circle[1]), 0.95*outer_circle[2])
            edges[~mask] = 0
            blobs = utils.get_blobs(edges)
            circles = []

            for blob in blobs:
                blob_x = blob[0]
                blob_y = blob[1]

                if len(blob_x) < 200:
                    split_number = 1
                else :
                    split_number = len(blob_x) // 200
                length = len(blob_x) // split_number

                for i in range(split_number):
                    if i == split_number-1: #last split
                        #max index (not included)
                        max_index = len(blob_x)
                    else:
                        max_index = (i+1) * length
                    x_temp, y_temp, r_temp = circledetection.fast_ols_circle_fit(blob_x[i * length:max_index], blob_y[i * length:max_index])
                    if not (x_temp is None or y_temp is None or r_temp is None):
                        if x_temp >= 0 and x_temp < edges.shape[1] and y_temp >= 0 and y_temp < edges.shape[0]:
                            circles.append((x_temp, y_temp, r_temp, len(blob_x[i * length:max_index]), blob))

            remaining_circles = circledetection.outliers_elimination_votes(edges.shape[0], edges.shape[1], circles, 8)

            if len(remaining_circles) == 0:
                continue

            blob_x = [x for circle in remaining_circles for x in circle[4][0]]
            blob_y = [y for circle in remaining_circles for y in circle[4][1]]

            x, y, r, cook_d = circledetection.fast_ols_circle_cook(blob_x, blob_y)

            if x is None or y is None or r is None:
                continue
        
            # Sort everything by Cook's distance
            # temp = [[x, y, c] for c, x, y in sorted(zip(cook_d, blob_x, blob_y), reverse=True)]
            # blob_x, blob_y, cook_d = np.transpose(temp)
            # blob_x = blob_x.astype("int")
            # blob_y = blob_y.astype("int")

            cook_threshold = 0.0017

            # Cut with a threshold
            temp = [[x, y, c] for c, x, y in zip(cook_d, blob_x, blob_y) if c < cook_threshold]
            blob_x, blob_y, cook_d = np.transpose(temp)
            blob_x = blob_x.astype("int")
            blob_y = blob_y.astype("int")

            x, y, r = circledetection.fast_ols_circle_fit(blob_x, blob_y)
    delta = time.process_time() - t
    print('least squares regression cook: ' + str(delta / num_iterations))


if __name__ == '__main__':
    # TIMES
    # hough - votes
    #
    # RESULTS
    # mean - variance

    # ols_test(5)
    # cook_test(5)

    # outer_circle_test(10)
    inner_circle_test(5)