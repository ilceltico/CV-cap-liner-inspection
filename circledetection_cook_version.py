import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import random
import statsmodels.api as sm

def leastSquaresCookVersion(x, y):
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


    indipVars = np.transpose(np.vstack((indip1, indip2)))
    #print(indipVars)

    # indip1 = sm.add_constant(indip1)
    # print(indip1)
    # indip2 = sm.add_constant(indip2)
    # print(indip2)
    # indip3 = sm.add_constant(indip3)
    # print(indip3)

    indipVars = sm.add_constant(indipVars)
    #print(indipVars)

    model = sm.OLS(dep,indipVars)
    results = model.fit()

    #print(results.summary())
    #print(results.params)

    xCenter = results.params[1]
    yCenter = results.params[2]
    radius = np.sqrt(results.params[0] + xCenter**2 + yCenter**2)
    #print("Center: (" + str(xCenter) + "," + str(yCenter) + "), radius: " + str(radius))

    cooks_distances = results.get_influence().summary_frame().cooks_d
    # print(sorted(cooks_distances, reverse=True))




    return xCenter, yCenter, radius, cooks_distances

if __name__ == '__main__':
    arrayX = np.array([2,3,1,2])
    arrayY = np.array([0,1,1,2])

    leastSquaresCookVersion(arrayX, arrayY)