import cv2 as cv
import os
import h5py
import numpy as np
import logging

def read_img(img_path):
    img = cv.imread(img_path)
    return img


def save_img(img, path):
    if img is None:
        print('error: no image')
    else:
        cv.imwrite(path, img)


def set_logger(logger_name):
    # Set up the logging handlers and loggers
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler(logger_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
