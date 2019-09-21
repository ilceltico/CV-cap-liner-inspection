# CV-cap-liner-inspection
Computer Vision project for the inspection of defects in plastic cap liners

## Technologies
The programming language used is Python, whereas the computer vision library used is OpenCV.

Required modules:
* numpy
* opencv
* statsmodel (optional, required only for circledetection.ols_circle_cook(), which isn't used.)

## Project structure
The repository contains a /caps folder, with test images for the code; a /tests folder, with older code, tests and test results; and a folder for organizational purposes.

Apart from these folders and the README.md, the root directory contains 4 Python files and a JSON file, for the configuration parameters. The Python files are:

* program.py, which contains the code for the main control flow and some internal functions for clarity and code reuse;
* loadconfiguration.py, which is the only module that needs to know the details of the configuration file. Its purpose is to load the file and initialize the global constants for execution;
* utils.py, which contains different useful functions that will be explained in detail in the report;
* circledetection.py, the richest file, in which all the non-library functions related to circle detection are contained.

The code is thoroughly commented, to ease comprehension, leaving to the report the theoretical explanations.

A code documentation is included: each function is described in detail, indicating the use of parameters, the structure of return values and the presence of possible exceptions.

## Use and Configuration
In order to configure the software, please set methods and parameters in config.json. Please don't change the json structure but work only on values.

For the outer circle detection ("outer" section) (i.e. cap mouth detection) the user can choose to use the Hough Transform ("method": "hough") or an ad hoc method ("method": "least_squares").
In both cases the user can set parameters for the method:

* in case of "method": "hough" the user can set "average_best_circles" and "radius_computation" parameters.
    * "average_best_circles" allows to specify how many circles (found with HoughCircles()) to average to compute the radius and center. Note that to avoid circle average and obtain the best circle found with the Hough Transform, the parameter "average_best_circle" must be equal to 1.
	* "radius_computation" allows the user to specify the method only for the radius computation. That means: take the center as an average of the best circles, and compute the radius subsequently. This is because, according to the official OpenCV doc, the radius computation in HoughCircles() function is not precise. In particular, if the value used is "border_distance" the radius is computed as the mean distance between edge points and the previously found center. Otherwise (i.e. "radius_computation": "mean_radius"), the method uses the previously found radius.
* in case of "method": "least_squares" the user can set "circle_generation" equal to "mean" or "interpolation".

For the inner circle detection ("inner" section) (i.e. liner border detection) the user can choose again to use the Hough Transform ("method": "hough") or an ad hoc method ("method": "least_squares").
Also here, the user can set parameters for both the methods:
* in case of "method": "hough" the user can select which kind of image will be passed to the Hough Transform ("canny precision"): 
    * "canny_precision": "precise" => the program will uses the Canny's edge detection result with L2 gradients to HoughCircles().
	* "canny_precision": "normal" => the program will uses a blurred image with a Gaussian filter as input to HoughCircles().

In both cases, the user can also specify the number of circles the software will average to obtain the final one (1 - i.e. return the first Hough cirlce - or N - i.e. the circle returned is the mean between the best N found by Hough, with N positive integer > 1).
* in case of "method": "least_squares" the user can specify if the blobs found should be splitted or not (true or false), which kind of outliers elimination perform ("mean" or "votes"), the minimum blob dimension and the circle generation strategy.

Please refers to the report for a detailed description of the methods.

## config.json structure

Please note that this is not a valid configuration file. It only shows the structure of the configuration file. It aims to to clarify and show all the possibilities described in "Use and Configuration" section.

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
                    "circle_generation": "mean" | "interpolation"
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
                    "split_blobs": true,
                    "min_blob_dim": 200,
                    "outliers_elimination_type": "mean" | "votes",
                    "circle_generation": "mean" | "interpolation" | "interpolation_cook"
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
                    "circle_generation": "mean"
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
                    "split_blobs": false,
                    "min_blob_dim": 200,
                    "outliers_elimination_type": "votes",
                    "circle_generation": "mean"
                }
            }
        }
    },
    "defect_detection":{

    }
}
```