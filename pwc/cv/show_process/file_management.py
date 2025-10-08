import os
import cv2 as cv
import numpy as np
import logging



def set_logger(logger_name=__name__, log_path='error.log'):
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

