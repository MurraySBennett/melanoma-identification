import os 
import pandas as pd
import numpy as np
import cv2 as cv
import glob
from pprint import pprint
import matplotlib.pyplot as plt
from tk import *
from os import path

from colourBalance import *
cb = colour_balance()

n_images = 2
##### --> set paths and filenames
home_path = path.join(path.expanduser("~/win_home/"), "melanoma-identification")
paths = dict(
        home=home_path,
        images=path.join(home_path, "images", "resized"),
        masks=path.join(home_path, "images", "segmented", "masks"),
        segmented=path.join(home_path, "images", "segmented", "images"),
        data=path.join(home_path, "images", "metadata")
        )

image_paths = glob.glob(paths["images"] + "/*.JPG")
image_ids = [i.split("/")[-1] for i in image_paths]

##### --> define functions

def display(img, labels):
    plt.figure(figsize=(len(img*2), 3))
    for i in range(len(img)):
        plt.subplot(1, len(img), i+1)
        plt.imshow(img[i])
        plt.axis('off')
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()

def get_shape_factor(contour):
    # f = 4piA / P**2
    a = cv.contourArea(contour)
    p = cv.arcLength(contour, True)
    f = np.divide(4 * np.pi * a, p**2) 
    return f


data = pd.read_csv(paths["data"] + "/metadata.csv")

data["image_raw"] = [cv.imread(path.join(paths["images"], i + ".JPG")) for i in
        data["isic_id"]]
data["image_proc"] = [process_img(i) for i in data["image_raw"]]
data["image_masks"], data["contours"], data["contoured_image"] = map(list, zip(*[hsv_channel(i) for i in
    data["image_proc"]]))
data["image_masked"] = [cv.bitwise_and(data["image_proc"][i], data["image_proc"][i],
    mask=data["image_masks"][i].astype(np.uint8)) for i in
    range(len(data["image_proc"]))]
shape_factor = [get_shape_factor(c) for c in contours]


# for i in range(n_images):
#     img = [image_proc[i], contoured_image[i], image_masked[i]]
#     labels = ["Raw", "contour", "mask"]
#     display(img, labels)


