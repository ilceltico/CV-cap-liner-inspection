import json

HARALICK_THRESHOLD = 200

OUTER_METHOD = ''
OUTER_LEAST_SQUARES_CIRCLE_GENERATION = ''

INNER_METHOD = ''
INNER_HOUGH_IMAGE = ''
INNER_HOUGH_NUMBER_AVG = ''
INNER_LEAST_SQUARES_SPLIT = ''
INNER_LEAST_SQUARES_SPLITTED_NUMBER = 1
INNER_LEAST_SQUARES_OUTLIERS_TYPE = ''
INNER_LEAST_SQUARES_CIRCLE_GENERATION = ''

def parse_json():
    global OUTER_METHOD
    global OUTER_LEAST_SQUARES_CIRCLE_GENERATION
    global INNER_METHOD
    global INNER_HOUGH_IMAGE
    global INNER_HOUGH_NUMBER_AVG
    global INNER_LEAST_SQUARES_SPLIT
    global INNER_LEAST_SQUARES_SPLITTED_NUMBER
    global INNER_LEAST_SQUARES_OUTLIERS_TYPE
    global INNER_LEAST_SQUARES_CIRCLE_GENERATION

    with open('config.json') as config_file:
        json_config = json.load(config_file)
        config = json_config['circle_detection']
        outer = config['outer']
        OUTER_METHOD = outer['method']
        if not OUTER_METHOD in ['hough', 'least_squares']:
            print('Configuration error. See README.md to configure the software properly.')
            #return None
            raise SystemExit(0)

        outer_parameters = outer['parameters'][OUTER_METHOD]
        if OUTER_METHOD == 'least_squares':
            OUTER_LEAST_SQUARES_CIRCLE_GENERATION = outer_parameters['circle_generation']
            if not OUTER_LEAST_SQUARES_CIRCLE_GENERATION in ['mean', 'interpolation']:
                print('Configuration error. See README.md to configure the software properly.')
                #return None
                raise SystemExit(0)

        inner = config['inner']
        INNER_METHOD = inner['method']
        if not INNER_METHOD in ['hough', 'least_squares']:
            print('Configuration error. See README.md to configure the software properly.')
            #return None
            raise SystemExit(0)

        inner_parameters = inner['parameters'][INNER_METHOD]
        if INNER_METHOD == 'hough':
            INNER_HOUGH_IMAGE = inner_parameters['image_to_hough']
            INNER_HOUGH_NUMBER_AVG = inner_parameters['number_of_circle_average']
            if not INNER_HOUGH_IMAGE in ['edges', 'gaussian'] or not isinstance(INNER_HOUGH_NUMBER_AVG, int) or (INNER_HOUGH_IMAGE == 'edges' and INNER_HOUGH_NUMBER_AVG < 1):
                print('Configuration error. See README.md to configure the software properly.')
                #return None
                raise SystemExit(0)
            
        else:
            INNER_LEAST_SQUARES_SPLIT = inner_parameters['split_blobs']
            INNER_LEAST_SQUARES_SPLITTED_NUMBER = inner_parameters['splitted_blobs_number']
            INNER_LEAST_SQUARES_OUTLIERS_TYPE = inner_parameters['outliers_elimination_type']
            INNER_LEAST_SQUARES_CIRCLE_GENERATION = inner_parameters['circle_generation']
            if not (isinstance(INNER_LEAST_SQUARES_SPLIT, bool)) or not (isinstance(INNER_LEAST_SQUARES_SPLITTED_NUMBER, int) and INNER_LEAST_SQUARES_SPLITTED_NUMBER > 0) or not (INNER_LEAST_SQUARES_OUTLIERS_TYPE in ['mean', 'bin']) or not (INNER_LEAST_SQUARES_CIRCLE_GENERATION in ['mean', 'interpolation', 'interpolation_cook']):
                print('Configuration error. See README.md to configure the software properly.')
                #return None
                raise SystemExit(0)
         
        #print('Configuration correctly loaded.')
        #return config