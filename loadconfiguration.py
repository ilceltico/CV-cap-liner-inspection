import json

HARALICK_THRESHOLD = 200

OUTER_METHOD = ''
OUTER_HOUGH_NUMBER_AVG = ''
OUTER_RADIUS_COMPUTATION = ''
OUTER_LEAST_SQUARES_CIRCLE_GENERATION = ''

INNER_METHOD = ''
INNER_CANNY_PRECISION = ''
INNER_HOUGH_NUMBER_AVG = ''
INNER_LEAST_SQUARES_SPLIT = ''
INNER_LEAST_SQUARES_MIN_BLOB_DIM = 1
INNER_LEAST_SQUARES_OUTLIERS_TYPE = ''
INNER_LEAST_SQUARES_CIRCLE_GENERATION = ''

def parse_json():
    global OUTER_METHOD
    global OUTER_HOUGH_NUMBER_AVG
    global OUTER_RADIUS_COMPUTATION
    global OUTER_LEAST_SQUARES_CIRCLE_GENERATION
    global INNER_METHOD
    global INNER_CANNY_PRECISION
    global INNER_HOUGH_NUMBER_AVG
    global INNER_LEAST_SQUARES_SPLIT
    global INNER_LEAST_SQUARES_MIN_BLOB_DIM
    global INNER_LEAST_SQUARES_OUTLIERS_TYPE
    global INNER_LEAST_SQUARES_CIRCLE_GENERATION

    with open('config.json') as config_file:
        json_config = json.load(config_file)
        config = json_config['circle_detection']
        outer = config['outer']
        OUTER_METHOD = outer['method']
        if not OUTER_METHOD in ['hough', 'least_squares']:
            print('Configuration error. See README.md to configure the software properly.')
            raise SystemExit(0)

        outer_parameters = outer['parameters'][OUTER_METHOD]
        if OUTER_METHOD == 'hough':
            OUTER_HOUGH_NUMBER_AVG = outer_parameters['average_best_circles']
            OUTER_RADIUS_COMPUTATION = outer_parameters['radius_computation']
            if not isinstance(OUTER_HOUGH_NUMBER_AVG, int) or not OUTER_HOUGH_NUMBER_AVG > 0 or not OUTER_RADIUS_COMPUTATION in ['mean_radius', 'border_distance']:
                print('Configuration error. See README.md to configure the software properly.')
                raise SystemExit(0)
        
        else:
            OUTER_LEAST_SQUARES_CIRCLE_GENERATION = outer_parameters['circle_generation']
            if not OUTER_LEAST_SQUARES_CIRCLE_GENERATION in ['mean', 'interpolation']:
                print('Configuration error. See README.md to configure the software properly.')
                raise SystemExit(0)

        inner = config['inner']
        INNER_METHOD = inner['method']
        if not INNER_METHOD in ['hough', 'least_squares']:
            print('Configuration error. See README.md to configure the software properly.')
            raise SystemExit(0)

        inner_parameters = inner['parameters'][INNER_METHOD]
        if INNER_METHOD == 'hough':
            INNER_CANNY_PRECISION = inner_parameters['canny_precision']
            INNER_HOUGH_NUMBER_AVG = inner_parameters['average_best_circles']
            if not INNER_CANNY_PRECISION in ['precise', 'normal'] or not isinstance(INNER_HOUGH_NUMBER_AVG, int) or INNER_HOUGH_NUMBER_AVG < 1:
                print('Configuration error. See README.md to configure the software properly.')
                raise SystemExit(0)
            
        else:
            INNER_LEAST_SQUARES_SPLIT = inner_parameters['split_blobs']
            INNER_LEAST_SQUARES_MIN_BLOB_DIM = inner_parameters['min_blob_dim']
            INNER_LEAST_SQUARES_OUTLIERS_TYPE = inner_parameters['outliers_elimination_type']
            INNER_LEAST_SQUARES_CIRCLE_GENERATION = inner_parameters['circle_generation']
            if not (isinstance(INNER_LEAST_SQUARES_SPLIT, bool)) or not (isinstance(INNER_LEAST_SQUARES_MIN_BLOB_DIM, int) and INNER_LEAST_SQUARES_MIN_BLOB_DIM > 0) or not (INNER_LEAST_SQUARES_OUTLIERS_TYPE in ['mean', 'votes']) or not (INNER_LEAST_SQUARES_CIRCLE_GENERATION in ['mean', 'interpolation', 'interpolation_cook']):
                print('Configuration error. See README.md to configure the software properly.')
                raise SystemExit(0)