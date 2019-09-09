import json

HARALICK_THRESHOLD = 200

ERROR_STRING = "Configuration error. See README.md to configure the software properly."

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
            raise SystemExit(ERROR_STRING + "\ncircle_detection -> outer -> method")

        outer_parameters = outer['parameters'][OUTER_METHOD]
        if OUTER_METHOD == 'hough':
            OUTER_HOUGH_NUMBER_AVG = outer_parameters['average_best_circles']
            OUTER_RADIUS_COMPUTATION = outer_parameters['radius_computation']
            if not isinstance(OUTER_HOUGH_NUMBER_AVG, int) or OUTER_HOUGH_NUMBER_AVG < 1:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> outer -> parameters -> hough -> average_best_circles")
            if not OUTER_RADIUS_COMPUTATION in ['mean_radius', 'border_distance']:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> outer -> parameters -> hough -> radius_computation")
        
        else:
            OUTER_LEAST_SQUARES_CIRCLE_GENERATION = outer_parameters['circle_generation']
            if not OUTER_LEAST_SQUARES_CIRCLE_GENERATION in ['mean', 'interpolation']:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> outer -> parameters -> least_squares -> circle_generation")

        inner = config['inner']
        INNER_METHOD = inner['method']
        if not INNER_METHOD in ['hough', 'least_squares']:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> method")

        inner_parameters = inner['parameters'][INNER_METHOD]
        if INNER_METHOD == 'hough':
            INNER_CANNY_PRECISION = inner_parameters['canny_precision']
            INNER_HOUGH_NUMBER_AVG = inner_parameters['average_best_circles']
            if not INNER_CANNY_PRECISION in ['precise', 'normal']:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> parameters -> hough -> canny_precision")
            if not isinstance(INNER_HOUGH_NUMBER_AVG, int) or INNER_HOUGH_NUMBER_AVG < 1:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> parameters -> hough -> average_best_circles")
            
        else:
            INNER_LEAST_SQUARES_SPLIT = inner_parameters['split_blobs']
            INNER_LEAST_SQUARES_MIN_BLOB_DIM = inner_parameters['min_blob_dim']
            INNER_LEAST_SQUARES_OUTLIERS_TYPE = inner_parameters['outliers_elimination_type']
            INNER_LEAST_SQUARES_CIRCLE_GENERATION = inner_parameters['circle_generation']
            if not (isinstance(INNER_LEAST_SQUARES_SPLIT, bool)):
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> parameters -> least_squares -> split_blobs")
            if not isinstance(INNER_LEAST_SQUARES_MIN_BLOB_DIM, int) or INNER_LEAST_SQUARES_MIN_BLOB_DIM < 1:
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> parameters -> least_squares -> min_blob_dim")
            if not (INNER_LEAST_SQUARES_OUTLIERS_TYPE in ['mean', 'votes']):
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> parameters -> least_squares -> outliers_elimination_type")
            if not (INNER_LEAST_SQUARES_CIRCLE_GENERATION in ['mean', 'interpolation', 'interpolation_cook']):
                raise SystemExit(ERROR_STRING + "\ncircle_detection -> inner -> parameters -> least_squares -> circle_generation")