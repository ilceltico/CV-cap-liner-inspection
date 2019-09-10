# CV-cap-liner-inspection
Computer Vision project for the inspection of defects in plastic cap liners

To configure the software, please set methods and parameters in config.json. Please don't change the json structure but change only the values.

For the outer circle detection ("outer" section) (i.e. cap mouth detection) the user can choose to use the Hough Transform ("method": "hough") or an ad hoc method ("least_squares").
In both cases the user can also set parameters for the method:
	- in case of "hough" method, the user can set "average_best_circles" and "radius_computation" parameters. The first one allows to specify how many circles (found with
	HoughCircles()) to average to compute the radius and center. The second one allows the user to specify the method only for the radius computation. 
	Note that in order to avoid circle average the parameter "average_best_circle" must be put equal to 1.
	- in case of "least_squares" method, the user can set "circle_generation" as "mean" or "interpolation": the circle will be obtained from the ones found respectively as mean
	circle or it will be interpolated.

For the inner circle detection ("inner" section) (i.e. liner border detection) the user can choose again to use the Hough Transform ("method": "hough") or the method "least_squares".
In case of "method": "hough" he can select which kind of image will be passed to the Hough Transform ("canny precision"): with "precise" selection the method will pass the Canny's 
edge detection result with L2 gradients to HoughCircles(), with "normal" the method will use a blurred image with a Gaussian filter. In both cases, the user can also specify the 
number of circles the software will average to obtain the final one (1 - i.e. return the first Hough cirlce - or N - i.e. the circle returned is the mean between the best N found 
by Hough, with N positive integer > 1).
In case of "method": "least_squares" the user can specify if the blobs found should be splitted or not (true or false), which kind of outliers elimination perform ("mean" or "votes"), 
the minimum blob dimension and the circle generation strategy.

Required modules:
-numpy
-opencv
-statsmodel (optional, required only for circledetection.ols_circle_cook(), which isn't used.)

Please refers to the report for a detailed description of the methods.


config.json structure:
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
					"canny_precision": ["precise", "normal"]
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

default configuration:
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