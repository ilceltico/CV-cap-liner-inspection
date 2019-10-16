# CV-cap-liner-inspection
Computer Vision project for the inspection of defects in plastic cap liners

## Technologies
The programming language used is Python, whereas the computer vision library used is OpenCV.

Required modules:
* numpy
* opencv
* statsmodels (optional, required only for circledetection.ols_circle_cook(), which isn't used.)

## Project structure
The repository contains a [/caps](https://github.com/ilceltico/CV-cap-liner-inspection/tree/master/caps) folder, with test images for the code; a [/tests](https://github.com/ilceltico/CV-cap-liner-inspection/tree/master/tests) folder, with older code, tests and test results; and a folder for organizational purposes.

Apart from these folders and the README.md, the root directory contains 5 Python files and a JSON file, for the configuration parameters. The Python files are:

* program.py, which contains the code for the main control flow and some internal functions for clarity and code reuse;
* loadconfiguration.py, which is the only module that needs to know the details of the configuration file. Its purpose is to load the file and initialize the global constants for execution;
* utils.py, which contains different useful functions that will be explained in detail in the report;
* circledetection.py, the richest file, in which all the non-library functions related to circle detection are contained.
* performancetester.py, which can be executed to asses the performances of the algorithms.

A code documentation is included: each function is described in detail, indicating the use of parameters, the structure of return values and the presence of possible exceptions.

## Use and Configuration
In order to configure the software, please set methods and parameters in config.json. Don't change the JSON structure.

For the outer circle detection ("outer" section) (i.e. cap mouth detection) the user can choose to use the Hough Transform ("method": "hough") or an ad hoc regression method ("method": "least_squares").
In both cases the user can set parameters for the method:

* in case of "method": "hough" the user can set "average_best_circles" and "radius_computation" parameters.
    * "average_best_circles" allows to specify how many circles (found with HoughCircles()) to average to compute the radius and center. To avoid this process and obtain simply the best circle found with the Hough Transform, the parameter "average_best_circle" must be equal to 1.
	* "radius_computation" allows the user to specify the method only for the radius computation. This is because, according to the official OpenCV doc, the radius computation in the HoughCircles() function is not precise. In particular, if the value used is "border_distance" the radius is computed as the mean distance between edge points and the previously found center. Otherwise (i.e. "radius_computation": "mean_radius"), the method uses the previously found radius: the mean radius of the best circles.
* in case of "method": "least_squares" the user can set "circle_generation" equal to "mean" or "least_squares". The former computes the regression separately on the different connected edges (blobs), with the final circle being a weighted mean of them; the latter computes the regression on all the edges.

For the inner circle detection ("inner" section) (i.e. liner border detection) the user can choose again to use the Hough Transform ("method": "hough") or an ad hoc method ("method": "least_squares").
Also here, the user can set parameters for both the methods:
* in case of "method": "hough" the user can select which kind of image will be passed to the Hough Transform ("canny precision"): 
    * "canny_precision": "precise" => the program will use the Canny's edge detection with L2 norm formula for gradient's magnitude computation; "normal" => the program will use the Canny's edge detection with L1 norm formula for gradient's magnitude computation, which is less isotropic (i.e. less precise in finding the correct edges).
    * In both cases, the user can also specify the number of circles that the software will average out to obtain the final one ("average_best_circles": 1, return the first Hough circle; N, the circle returned is the mean between the best N found by Hough, with N positive integer > 1).
* in case of "method": "least_squares" the user can specify:
    * If the blobs found should be splitted or not ("split_blobs", true or false), to produce a more meaningful blob-outliers elimination.
    * The minimum blob dimension (in case of splitting): the splitting procedure will not produce blobs that are smaller than specified.
    * Which kind of blob-outliers elimination to perform, if any. "none": do not perform it; "mean": exclude that blobs that produce a circle too far away from the weighted average circle of all the blobs; "votes": voting process similar to the one found in the Hough Transform, with each blob voting for the circle that best fits it in a least squares sense and having as many votes as its own pixels.
    * The circle generation strategy ("mean", "least_squares" or "least_squares_cook"). The latter also computes the Cook's distance and eliminates pixel outliers whose Cook's distance surpasses a threshold.

Please refer to the report for a detailed description of the methods.

## config.json structure

Please note that this is not a valid configuration file. It only shows the structure of the configuration file. It aims to clarify and show all the possibilities described in the "Use and Configuration" section.

The "|" means that for the attribute the user can specify one of the possibilites. Remember also that the numbers must be always > 0. 

Please, refer to the next section ("Default structure") to have an example of configuration file. 

```
{
    "circle_detection":{
        "outer":{
            "method": "hough" | "least_squares",
            "parameters": {
                "hough": {
                    "average_best_circles": 3,
                    "radius_computation": "mean_radius" | "border_distance"
                },
                "least_squares": {
                    "circle_generation": "mean" | "least_squares"
                }
            }
        },
        "inner": {
            "method": "hough" | "least_squares",
            "parameters": {
                "hough": {
                    "canny_precision": "precise" | "normal",
                    "average_best_circles": 2
                },
                "least_squares": {
                    "split_blobs": true | false,
                    "min_blob_dim": 200,
                    "outliers_elimination_type": "none" | "mean" | "votes",
                    "circle_generation": "mean" | "least_squares" | "least_squares_cook"
                }
            }
        }
    },
    "defect_detection":{

    }
}
```

## Default configuration

```json
{
    "circle_detection":{
        "outer":{
            "method": "hough",
            "parameters":{
                "hough":{
                    "average_best_circles": 3,
                    "radius_computation": "border_distance"
                },
                "least_squares": {
                    "circle_generation": "least_squares"
                }
            }
        },
        "inner": {
            "method": "hough",
            "parameters": {
                "hough": {
                    "canny_precision": "precise",
                    "average_best_circles": 2
                },
                "least_squares": {
                    "split_blobs": true,
                    "min_blob_dim": 200,
                    "outliers_elimination_type": "votes",
                    "circle_generation": "least_squares"
                }
            }
        }
    },
    "defect_detection":{

    }
}
```