# CV-cap-liner-inspection
Computer Vision project for the inspection of defects in plastic cap liners

To configure the software, please set methods and parameters in config.json. Please don't change the json structure but change only the values.

For the outer circle detection ("outer" section) (i.e. cap mouth detection) the user can choose to use the Hough Transform ("method": "hough") or an ad hoc method ("least_squares").
In the latter case the user can also set the "circle_generation" parameter as "mean" or "interpolation": the circle will be obtained from the ones found respectively as mean circle or it will be interpolated.

For the inner circle detection ("inner" section) (i.e. liner border detection) the user can choose again to use the Hough Transform ("method": "hough") or the method "least_squares".
In case of "method": "hough" he can select which kind of image will be passed to the Hough Transform ("edges" or "gaussian"). In case of "edges" selection, he can also specify the 
number of circles the software will average to obtain the final circle (1 - i.e. return the first Hough cirlce - or 2 - i.e. the circle returned is the mean between the best two found 
by Hough).
In case of "method": "least_squares" the user can specify if the blobs found should be splitted or not (true or false), which kind of outliers elimination perform ("mean" or "bin") 
and the circle generation strategy. Note: you cannot use the combination "split":"true" with "outliers_elimination_type":"mean".

Required modules:
-numpy
-opencv
-statsmodel

Please refers to the relation for a detailed description of the methods.


config.json structure:
```json 
{
    "circle_detection":{
        "outer":{
            "method": ["hough", "least_squares"],
            "parameters":{
                "circle_generation": ["mean", "interpolation"]
            }
        },
        "inner": {
            "method": ["hough", "least_squares"],
            "parameters": {
                "hough": {
                    "image_to_hough": ["edges", "gaussian"],
                    "number_of_circle_average": 2
                },
                "least_squares": {
                    "split_blobs": true,
                    "outliers_elimination_type": ["mean", "bin"],
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
                "circle_generation": "mean"
            }
        },
        "inner": {
            "method": "hough",
            "parameters": {
                "hough": {
                    "image_to_hough": "edges",
                    "number_of_circle_average": 2
                },
                "least_squares": {
                    "split_blobs": false,
                    "outliers_elimination_type": "bin",
                    "circle_generation": "mean"
                }
            }
		}
    },
    "defect_detection":{

    }
}
```