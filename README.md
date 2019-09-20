# CV-cap-liner-inspection
Computer Vision project for the inspection of defects in plastic cap liners

## Use and Configuration
In order to configure the software, please set methods and parameters in config.json. Please don't change the json structure but work only on values.

For the outer circle detection ("outer" section) (i.e. cap mouth detection) the user can choose to use the Hough Transform ("method": "hough") or an ad hoc method ("method": "least_squares").
In both cases the user can set parameters for the method:

* in case of "method": "hough", the user can set "average_best_circles" and "radius_computation" parameters.
    * "average_best_circles" allows to specify how many circles (found with HoughCircles()) to average to compute the radius and center. Note that to avoid circle average and obtain the best circle found with the Hough Transform, the parameter "average_best_circle" must be put equal to 1.
	* "radius_computation" allows the user to specify the method only for the radius computation. That means: take the center as an average of the best circles, and compute the radius subsequently. This is because, according to the official OpenCV doc, the radius computation in its function is not precise. In particular, if the value used is "border_distance" the radius is computed as the mean distance between points and the previously found center. Otherwise (i.e. "radius_computation": "mean_radius"), the method uses the previously found radius.
* in case of "method": "least_squares", the user can set "circle_generation" equal to "mean" or "interpolation". The circle will be obtained from the edge points respectively as mean circle of the ones interpolated from blobs or it will be interpolated directly from all the edge points.

For the inner circle detection ("inner" section) (i.e. liner border detection) the user can choose again to use the Hough Transform ("method": "hough") or an ad hoc method ("method": "least_squares").
Also here, the user can set parameters for both the methods:
* in case of "method": "hough", the user can select which kind of image will be passed to the Hough Transform ("canny precision"): 
    * "canny_precision": "precise" => the program will the Canny's edge detection result with L2 gradients to HoughCircles();
	* "canny_precision": "normal" => the program will use a blurred image with a Gaussian filter.
In both cases, the user can also specify the number of circles the software will average to obtain the final one (1 - i.e. return the first Hough cirlce - or N - i.e. the circle returned is the mean between the best N found by Hough, with N positive integer > 1).
* in case of "method": "least_squares", the user can specify if the blobs found should be splitted or not (true or false), which kind of outliers elimination perform ("mean" or "votes"), the minimum blob dimension and the circle generation strategy.

Please refers to the report for a detailed description of the methods.

## Technologies
The programming language used is Python, whereas the computer vision libraryto be used is OpenCV.

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

The code is thoroughly commented, to ease comprehension, leaving to thisreport the theoretical explanations.

A code documentation is included: each function is described in detail, indicating the use of parameters, the structure of return values and the presence of possible exceptions.

Please refers to the report and the code comments for a detailed description of the source code.

## config.json structure

```json 
{
    "circle_detection":{
        "outer":{
            "method": ["hough", "least_squares"],
            "parameters": {
				"hough": {
					"average_best_circles": 3,
                    "radius_computation": ["mean_radius", "border_distance"]
				},
				"least_squares": {
					"circle_generation": ["mean", "interpolation"]
				}
            }
        },
        "inner": {
            "method": ["hough", "least_squares"],
            "parameters": {
                "hough": {
					"canny_precision": ["precise", "normal"],
                    "average_best_circles": 2
                },
                "least_squares": {
                    "split_blobs": true,
					"min_blob_dim": 200,
                    "outliers_elimination_type": ["mean", "votes"],
                    "circle_generation": ["mean", "interpolation", "interpolation_cook"]
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