# CV-cap-liner-inspection
Computer Vision project for the inspection of defects in plastic cap liners

config.json structure:
{
    "circle_detection":{
        "outer":{
            "method": ["hough", "least_squares"]
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

default configuration:
{
    "circle_detection":{
        "outer":{
            "method": "hough",
            "parameters":{
                "circle_generation": ["mean", "interpolation"]
            }
        },
        "inner": {
            "method": "hough",
            "parameters": {
                "hough": {
                    "image_to_hough": "gaussian",
                    "number_of_circle_average": 2
                },
                "least_squares": {
                    "split_blobs": true,
                    "outliers_elimination_type": "bin",
                    "circle_generation": ["mean", "interpolation", "interpolation_cook"]
                }
            }
		}
    },
    "defect_detection":{

    }
}