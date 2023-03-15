#!usr/bin/env python3 

import os 
import pandas as pd
import numpy as np
import cv2 as cv
import glob
from pprint import pprint

from os import path

from colourBalance import *
cb = colour_balance()
)
##### --> set paths and filenames
home_path = path.join(path.expanduser("~/win_home/"), "melanoma-identification")
paths = dict(
        home=home_path,
        images=path.join(home_path, "images", "resized"),
        masks=path.join(home_path, "images", "segmented", "masks"),
        segmented=path.join(home_path, "images", "segmented", "images")
        )        
images = glob.glob(paths["images"] + "/*.JPG")

##### --> define functions

def segment(img):
    gray, blkhat, thresh2, processed = process_img(img) 

    mask, contour, contoured_img, SV_conversion = hsv_channel(processed)
    masked_img = impose_mask(img, mask)

    return processed, masked_img, mask, contour, contoured_img, SV_conversion


