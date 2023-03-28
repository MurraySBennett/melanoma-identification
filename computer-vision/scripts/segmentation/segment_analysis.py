import numpy as np


def get_shape_factor(contour):
    """f = 4piA / P**2"""
    a = cv.contourArea(contour)
    p = cv.arcLength(contour, True)
    f = np.divide(4 * np.pi * a, p**2) 
    return f
