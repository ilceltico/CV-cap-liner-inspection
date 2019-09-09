import numpy as np
import math
import statsmodels.api as sm
import utils
import cv2
import time

def find_circle_ols(edges, min_blob_dim, outliers_elimination, final_computation_method, 
        oe_thresholds=(20,20), oe_bins_factor=8):
    """
    """
    # tic = time.process_time()

    blobs = utils.get_blobs(edges)
    circles = []

    # Find a circle fit for each blob
    if min_blob_dim == 0:
        for blob in blobs:
            x_temp, y_temp, r_temp = least_squares_circle_fit(blob[0], blob[1])
            if not (x_temp is None or y_temp is None or r_temp is None):
                if x_temp >= 0 and x_temp < edges.shape[1] and y_temp >= 0 and y_temp < edges.shape[0]:
                    circles.append((x_temp, y_temp, r_temp, len(blob[0]), blob))

    # Split the blobs and find a fit
    else:
        # coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for blob in blobs:
            blob_x = blob[0]
            blob_y = blob[1]

            if len(blob_x) < min_blob_dim:
                split_number = 1
            else :
                split_number = len(blob_x) // min_blob_dim
            length = len(blob_x) // split_number

            for i in range(split_number):
            #     img_copy = coloured_image.copy()
            #     color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            #     for j in range(0, length):
            #         if j+i*length < len(blobX):
            #             img_copy[blob_x[j+i*length]][blob_y[j+i*length]] = color

                if i == split_number-1: #last split
                    #max index (not included)
                    max_index = len(blob_x)
                else:
                    max_index = (i+1) * length
                x_temp, y_temp, r_temp = least_squares_circle_fit(blob_x[i * length:max_index], blob_y[i * length:max_index])
                if not (x_temp is None or y_temp is None or r_temp is None):
                    if x_temp >= 0 and x_temp < edges.shape[1] and y_temp >= 0 and y_temp < edges.shape[0]:
                        circles.append((x_temp, y_temp, r_temp, len(blob_x[i * length:max_index]), blob))

    # Eliminate circles that are too far away from the weighted mean
    if outliers_elimination == 'mean':
        remaining_circles = outliers_elimination_mean(circles, (20,20))
    
    # Eliminate outliers using a voting process
    elif outliers_elimination == 'votes':
        remaining_circles = outliers_elimination_votes(edges.shape[0], edges.shape[1], circles, oe_bins_factor) 
    
    # No outliers elimination
    elif outliers_elimination == None:
        remaining_circles = circles

    # Re-compute a weighted mean circle
    if final_computation_method == 'mean':
        weighted = [[circle[0] * circle[3], circle[1] * circle[3], circle[2] * circle[3], circle[3]] for circle in remaining_circles]
        sums = [sum(a) for a in zip(*weighted)]
        x, y, r, _ = [el/sums[3] for el in sums]

    else:
        blob_x = [x for circle in remaining_circles for x in circle[4][0]]
        blob_y = [y for circle in remaining_circles for y in circle[4][1]]

        # Delete single-point outliers by computing the Cook's distance
        if final_computation_method == 'interpolation_cook':
            x, y, r, cook_d = least_squares_circle_cook(blob_x, blob_y)

            if x is None or y is None or r is None:
                return x, y, r
            
            # Sort points by decreasing Cook's distance
            blob_x = [x for _, x in sorted(zip(cook_d, blob_x), reverse=True)]
            blob_y = [y for _, y in sorted(zip(cook_d, blob_y), reverse=True)]
            cook_d = sorted(cook_d, reverse=True)

            normalizedCooks = np.array(cook_d)/max(cook_d) * 200 + 55

            imgCook = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for i in range(0, len(blob_x)):
                imgCook[blob_y[i],blob_x[i]] = (normalizedCooks[i],100,0)

            cv2.imshow('imgCook', imgCook)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("Highest Cook's distance: " + str(cook_d[0]))

            cook_threshold = 0.003

            blob_x = [x for cook, x in zip(cook_d, blob_x) if cook < cook_threshold]
            blob_y = [y for cook, y in zip(cook_d, blob_y) if cook < cook_threshold]
            cook_d = [c for c in cook_d if c < cook_threshold]
            
            # elimination_fraction = 10

            # blob_x = blob_x[int(len(blob_x) / elimination_fraction):]
            # blob_y = blob_y[int(len(blob_y) / elimination_fraction):]

            # cook_d = cook_d[int(len(cook_d) / elimination_fraction):]

            normalizedCooks = np.array(cook_d)/max(cook_d) * 200 + 55
            imgCook = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for i in range(0, len(blob_x)):
                imgCook[blob_y[i],blob_x[i]] = (normalizedCooks[i],100,0)

            cv2.imshow('imgCook', imgCook)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # Interpolate by fitting again
        x, y, r = least_squares_circle_fit(blob_x, blob_y)

    # delta = time.process_time() - tic
    # print("Delta = " + str(delta))

    return x, y, r

        
def find_circles_hough(img, dp, min_dist, canny_th_high, accumulator_threshold, 
    min_radius, max_radius, no_circles_to_avg):

    # tic = time.process_time()

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, min_dist, param1=canny_th_high, param2=accumulator_threshold, 
        minRadius=min_radius, maxRadius=max_radius)

    # Compute the result as an average of the best circles
    number_of_circles = no_circles_to_avg
    if len(circles[0]) == 0:
        print("No circles found!")
        return None, None, None
    if len(circles[0]) < number_of_circles:
        number_of_circles = len(circles[0])
        print("WARNING: less than the specified amount of circles was found, using " + str(number_of_circles) + " circles only.")

    x = sum(circles[0][i][0] for i in range(number_of_circles)) / number_of_circles # try np.mean()
    y = sum(circles[0][i][1] for i in range(number_of_circles)) / number_of_circles
    r = sum(circles[0][i][2] for i in range(number_of_circles)) / number_of_circles

    # delta = time.process_time() - tic
    # print("Delta = " + str(delta))

    return x, y, r


def least_squares_circle_fit(x, y):
    """
    Interpolate the points with a circle.

    Parameters:
        x: list of x coordinates.
        y: list of y coordinates.

    Return:
        Center x coordinate, center y coordinate and radius
    """

    num_points = len(x)

    if num_points == 0:
        return None, None, None
    
    x_ = np.sum(x) / num_points
    y_ = np.sum(y) / num_points

    u = x - x_
    v = y - y_
    
    #cache version
    usquare = np.square(u)
    vsquare = np.square(v)
    suu = np.sum(usquare)
    suv = np.sum(np.multiply(u, v))
    svv = np.sum(vsquare)
    suuu = np.sum(np.multiply(u, usquare))
    svvv = np.sum(np.multiply(v, vsquare))
    suvv = np.sum(np.multiply(u, vsquare))
    svuu = np.sum(np.multiply(v, usquare))
    
    #ucvc = np.linalg.solve(np.array([[suu, suv], [suv, svv]]), np.array([(suuu+suvv)/2, (svvv+svuu)/2]))
    # suu * uc + suv * vc = (suuu+suvv)/2
    # suv * uv + svv * vc = (svvv+svuu)/2

    # ax + by = c
    # dx + ey = f

    # ax + by - a(dx+ey)/d = c - af/d
    # ax + by - ax - aey/d = c - af/d
    # y(b - ae/d) = c - af/d
    # y = (c - af/d)/(b - ae/d) = (dc - af)/(bd - ae)

    # x = (f - ey)/d
    #a = suu
    #b = suv
    #c = (suuu+suvv)/2
    #d = suv
    #e = svv
    #f = (svvv+svuu)/2

    if suv*suv - suu*svv == 0 or suv == 0:
        return None, None, None

    vc = (suv*(suuu+suvv)/2 - suu*(svvv+svuu)/2)/(suv*suv - suu*svv)
    uc = ((svvv+svuu)/2 - svv*vc)/suv

    xc = uc + x_
    yc = vc + y_
    alfa = uc*uc + vc*vc + (suu+svv)/num_points
    r = math.sqrt(alfa)

    return xc, yc, r

def least_squares_circle_cook(x, y):
    """
    Interpolate the points with a circle.

    Parameters:
        x: list of x coordinates.
        y: list of y coordinates.

    Return:
        Center x coordinate, center y coordinate and radius
    """
    
    try:
        x = np.array(x)
        y = np.array(y)
        indip1 = 2*x
        #print(indip1)
        indip2 = 2*y
        #print(indip2)
        dep = x**2 + y**2
        #print(dep)


        indip_vars = np.transpose(np.vstack((indip1, indip2)))
        #print(indipVars)

        indip_vars = sm.add_constant(indip_vars)
        #print(indipVars)

        model = sm.OLS(dep, indip_vars)
        results = model.fit()

        #print(results.summary())
        #print(results.params)

        x_center = results.params[1]
        y_center = results.params[2]
        radius = np.sqrt(results.params[0] + x_center**2 + y_center**2)
        #print("Center: (" + str(xCenter) + "," + str(yCenter) + "), radius: " + str(radius))

        cooks_distances = results.get_influence().summary_frame().cooks_d
        # print(sorted(cooks_distances, reverse=True))

        return x_center, y_center, radius, cooks_distances
    except:
        return None, None, None, []

def outliers_elimination_mean(circles, thresholds):
    """
    Eliminate outliers circles based on the distance from the mean circles.

    Parameters:
        circles: list of center x coordinate, center y coordinate, radius and number of points (eventually also the blob)
        thresholds: tuple of max distance from the center and difference from the radius of the mean circle.

    Return:
        List of remaining circles.
    """

    # circles: list of tuple (center x, center y, radius, number of pixels)
    #thresholds: tuple of thresholds ((center x, center y), radius)

    if len(circles) == 0:
        return []

    weighted = [[circle[0] * circle[3], circle[1] * circle[3], circle[2] * circle[3], circle[3]] for circle in circles]
    sums = [sum(a) for a in zip(*weighted)]
    mean_circle = [el/sums[3] for el in sums]

    remaining_circles = [circle for circle in circles if math.sqrt((circle[0] - mean_circle[0]) ** 2 + (circle[1] - mean_circle[1]) ** 2) <= thresholds[0] and abs(circle[2] - mean_circle[2]) <= thresholds[1]]

    return remaining_circles

def outliers_elimination_votes(img_height, img_width, circles, resolution_factor):
    """
    Eliminate outliers circles based on similarity.

    Parameters:
        img_height: int.
        imag_width: int.
        circles: list of center x coordinate, center y coordinate, radius and number of points (eventually also the blob)
        resolution_factor: 1 means that the functions uses as many bins as pixels, 2 means half, etc.
    Return:
        List of remaining circles.
    """

    if len(circles) == 0:
        return []

    # If there are enough circles, optimize by using a Hough-like 3D memory structure
    if len(circles) > 1000:
        bins = (math.ceil(img_height/resolution_factor), math.ceil(img_width/resolution_factor), math.ceil(max([img_height, img_width])/resolution_factor))
        votes_bins = np.zeros((bins[0], bins[1], bins[2]))
        circle_bins = [[[[] for _ in range(bins[2])] for _ in range(bins[1])] for _ in range(bins[0])]

        for circle in circles:
            row_bin = np.round(circle[0]).astype("int") // resolution_factor
            col_bin = np.round(circle[1]).astype("int") // resolution_factor
            r_bin = np.round(circle[2]).astype("int") // resolution_factor

            votes_bins[row_bin][col_bin][r_bin] += circle[3]
            circle_bins[row_bin][col_bin][r_bin].append(circle)

        maximum = np.unravel_index(np.argmax(votes_bins, axis=None), votes_bins.shape)
        remaining_circles = circle_bins[maximum[0]][maximum[1]][maximum[2]]

    # If not, use a simple dictionary
    else:
        votes_bins = {}
        
        for circle in circles:
            row_bin = np.round(circle[0]).astype("int") // resolution_factor
            col_bin = np.round(circle[1]).astype("int") // resolution_factor
            r_bin = np.round(circle[2]).astype("int") // resolution_factor

            key = (row_bin,col_bin,r_bin)

            if not key in votes_bins:
                votes_bins[key] = [circle[3], [circle]]
            else:
                value = votes_bins[key]
                value[0] += circle[3]
                value[1].append(circle)
                votes_bins[key] = value

        remaining_circles = max(list(votes_bins.values()), key = lambda x:x[0])[1]

    return remaining_circles
