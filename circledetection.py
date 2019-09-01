import numpy as np
import math
import statsmodels.api as sm

def least_squares_circle_fit(x, y):
    """
    Interpolate the points with a circle.

    Parameters:
        x: list of x coordinates.
        y: list of y coordinates.

    Return:
        Tuple containing center coordinates and radius
    """

    num_points = len(x)

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
        return float('NaN'), float('NaN'), float('NaN')

    vc = (suv*(suuu+suvv)/2 - suu*(svvv+svuu)/2)/(suv*suv - suu*svv)
    uc = ((svvv+svuu)/2 - svv*vc)/suv

    xc = uc + x_
    yc = vc + y_
    alfa = uc*uc + vc*vc + (suu+svv)/numPoints
    r = math.sqrt(alfa)

    return xc, yc, r

def least_squares_circle_cook(x, y):
    """
    Interpolate the points with a circle.

    Parameters:
        x: list of x coordinates.
        y: list of y coordinates.

    Return:
        Tuple containing center coordinates, radius and cook distance list
    """
    
    x = np.array(x)
    y = np.array(y)
    indip1 = 2*x
    #print(indip1)
    indip2 = 2*y
    #print(indip2)
    indip3 = np.ones(len(x), dtype=int)
    #print(indip3)
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


def outliers_elimination(circles, thresholds):
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

    circles_remaining = [(circle[0], circle[1], circle[2], circle[3]) for circle in circles if math.sqrt((circle[0] - mean_circle[0]) ** 2 + (circle[1] - mean_circle[1]) ** 2) <= thresholds[0] and abs(circle[2] - mean_circle[2]) <= thresholds[1]]

    # if len(circles[0]) == 4:
    #     weighted = [[x * n, y * n, r * n, n] for x, y, r, n in circles]
    #     sums = [sum(a) for a in zip(*weighted)]
    #     mean_circle = [el/sums[3] for el in sums]

    #     circles_remaining = [(x, y, r, n) for x, y, r, n in circles if math.sqrt((x - mean_circle[0]) ** 2 + (y - mean_circle[1]) ** 2) <= thresholds[0] and abs(r - mean_circle[2]) <= thresholds[1]]
    # else:
    #     weighted = [[x * n, y * n, r * n, n] for x, y, r, n, _ in circles]
    #     sums = [sum(a) for a in zip(*weighted)]
    #     mean_circle = [el/sums[3] for el in sums]

    #     circles_remaining = [(x, y, r, n, blob) for x, y, r, n, blob in circles if math.sqrt((x - mean_circle[0]) ** 2 + (y - mean_circle[1]) ** 2) <= thresholds[0] and abs(r - mean_circle[2]) <= thresholds[1]]

    return circles_remaining

# # metodo uguale a quello sopra ma prende circle con in più i blobs per fare poi interpolazione
# # possiamo decidere di usare sempre questo e buttare via quello sopra
# def outliers_elimination_blobs(circles, thresholds):
#     """
#     """

#     # circles: list of tuple (center x, center y, radius, number of pixels)
#     #thresholds: tuple of thresholds ((center x, center y), radius)

#     if len(circles) == 0:
#         return None, None, None

#     weighted = [[x * n, y * n, r * n, n] for x, y, r, n, _ in circles]
#     sums = [sum(a) for a in zip(*weighted)]
#     mean_circle = [el/sums[3] for el in sums]

#     circles_remaining = [(x, y, r, n, blob) for x, y, r, n, blob in circles if math.sqrt((x - mean_circle[0]) ** 2 + (y - mean_circle[1]) ** 2) <= thresholds[0] and abs(r - mean_circle[2]) <= thresholds[1]]

#     if len(circles_remaining) > 0:
#         return circles_remaining
#     else:
#         return None, None, None

def outliers_elimination_with_bins(img_height, img_width, circles, bins):
    """
    Eliminate outliers circles based on similarity.

    Parameters:
        img_height: int.
        imag_width: int.
        circles: list of center x coordinate, center y coordinate, radius and number of points (eventually also the blob)
        thresholds: tuple of numbers of bins on rows, columns and radius
    Return:
        List of remaining circles.
    """

    if len(circles) == 0:
        return []

    # img_shape: rows and columns
    # circles: list of tuple (center x, center y, radius, number of pixel)
    # bins: tuple of number of bins ((axis x, axis y), radius)

    votes_bins = np.zeros((bins[0][0], bins[0][1], bins[1]))
    circle_bins = [[[[] for _ in range(bins[1])] for _ in range(bins[0][1])] for _ in range(bins[0][0])]

    bin_shape_rows = img_height // bins[0]
    bin_shape_cols = img_width // bins[1]
    bin_shape_r = img_height // 2 // bins[2]

    for circle in circles:
        row_bin = np.round(circle[0]).astype("int") // bin_shape_rows
        col_bin = np.round(circle[1]).astype("int") // bin_shape_cols
        r_bin = np.round(circle[2]).astype("int") // bin_shape_r

        votes_bins[row_bin][col_bin][r_bin] += circle[3]
        circle_bins[row_bin][col_bin][r_bin].append(circle)

    maximum = np.unravel_index(np.argmax(votes_bins, axis=None), votes_bins.shape)
    remaining_circles = circle_bins[maximum[0]][maximum[1]][maximum[2]]

    # TO USE THIS PART THE USER NEED TO PASS THE BLOBS TOGETHER WITH THE CIRCLES

    # blob_x = [x for circle in remaining_circles for x in circle[4][0]]
    # blob_y = [y for circle in remaining_circles for y in circle[4][1]]

    # x, y, r = circledetection.leastSquaresCircleFitCached(blob_x, blob_y)

    # return x, y, r

    return remaining_circles
