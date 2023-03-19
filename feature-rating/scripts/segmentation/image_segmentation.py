import os 
import pandas as pd
import numpy as np
import cv2 as cv
import glob
from pprint import pprint
import matplotlib.pyplot as plt
from tk import *
from os import path
from skimage.segmentation import chan_vese
from dca_removal import * 

n_images = 5
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


def process_img(img):
    # It absolutely pins a few test iamges with a resizing to 192,192. So
    # frustrating.
    # img = cv.resize(img, (256, 192))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Kernel for the morphological filtering
    kernel = cv.getStructuringElement(1, (5, 5))
    # Perform blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    closing = cv.morphologyEx(blackhat, cv.MORPH_CLOSE, kernel)
    bhg = cv.GaussianBlur(closing, (3, 3), cv.BORDER_DEFAULT)
    # intensify the hair countours in preparation for the inpainting algorithm
    ret, thresh2 = cv.threshold(bhg, 1, 255, cv.THRESH_BINARY)
    # inpaint the original image depending on the mask
    processed  = cv.inpaint(img, thresh2, 6, cv.INPAINT_TELEA)
    return processed 



def grabcut_segment(img):
    # github.com/fitushar/skin-lesion-segmentation...
    median = cv.medianBlur(img, 5)
    Z = np.float32(median.reshape((-1, 3)))
    
    # kmeans clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((img.shape))
    
    ###### adaptive histogram equalisation -- contrast limited (clahe)
    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    hsv = cv.cvtColor(kmeans_img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = clahe.apply(h)
    s = clahe.apply(s)
    v = clahe.apply(v)

    lab = cv.merge((h, s, v))
    
    enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    
    ###### otsu thresholding
    hsv_img = cv.cvtColor(enhanced, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_img)
    converted_img = S + V
    blur = cv.GaussianBlur(converted_img, (5,5), 0)
    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    canny_otsu = cv.Canny(otsu, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    canny_otsu = cv.dilate(canny_otsu, kernel, iterations=1)

    contours, hierarchy = cv.findContours(canny_otsu, cv.RETR_TREE,
            cv.CHAIN_APPROX_NONE)
    h, w, d = img.shape
    new_contours=[]
    # for c in range(len(contours)):
    #     if np.any(contours[c] == 0) or np.any(contours[c] == w-1):
    #         continue
    #     new_contours.append(contours[c])

    # cntsSorted = sorted(new_contours, key=lambda x: cv.contourArea(x))
    # largest_contour = cntsSorted[-1]
    if len(contours) != 0:
        largest_contour = max(contours, key=cv.contourArea)
    otsu_mask = np.zeros((h, w)).astype(np.uint8)
    cv.fillPoly(otsu_mask, pts=[largest_contour], color=(255, 255, 255))



    ###### mask generation
    hsv = cv.cvtColor(enhanced, cv.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([100, 255, 255])
    mask_g = cv.inRange(hsv, lower_green, upper_green)

    ret, inv_mask = cv.threshold(mask_g, 127, 255, cv.THRESH_BINARY_INV)
    res = cv.bitwise_and(img, img, mask=mask_g)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # inv_mask is 384x512 of 0s or 255. 
    mask_threshold = inv_mask.shape[0] * inv_mask.shape[1] * 255 * 0.75
    if (np.sum(inv_mask[:]) < mask_threshold):
        new_mask = inv_mask
        mask[new_mask == 0] = 0
        mask[new_mask == 255] = 1
        dim = cv.grabCut(img, mask, None, bgdModel, fgdModel, 5,
                cv.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        grabcut_img = img * mask2[:, :, np.newaxis]
    else: #### GRABCUT
    # initialise rectanlge based on image dimensions
        s = (img.shape[0] / 10, img.shape[1] / 10)
        rect = (int(s[0]), int(s[1]), int(img.shape[0] - (3/10) * s[0]), int(img.shape[1] - s[1]))
        dim = cv.grabCut(enhanced, mask, rect, bgdModel, fgdModel, 10,
            cv.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        grabcut_img = img*mask2[:, :, np.newaxis]
    

    ###### binarisation
    img_mask = cv.medianBlur(grabcut_img, 5)
    ret, segmented_mask = cv.threshold(img_mask, 0, 255, cv.THRESH_BINARY)
    


    display(
        [img, kmeans_img, lab, hsv, inv_mask, res, grabcut_img, segmented_mask,otsu_mask],
        ["original", "kmeans", "lab", "hsv", "inv_mask", "res", "grabcut_img","segmented_mask", "otsu_mask"]
    )


def segment(img):
    contoured_image = img.copy()  # copy() so that you are not drawing over the original image
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_img)
    converted_img = S + V
    blur = cv.GaussianBlur(converted_img, (5, 5), 0)
    

    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # I don't think this adds anything
    # ch_v = chan_vese(otsu, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
    #                max_num_iter=200, dt=0.5, init_level_set="disk",
    #                extended_output=False)

    canny_h = cv.Canny(H, 50, 120)
    canny_s = cv.Canny(S, 100, 200)
    canny_v = cv.Canny(V, 50, 100)
    canny_otsu = cv.Canny(otsu, 100, 200)
    
    processed = canny_otsu
    # Dilate img to 'close' spaces -- connect canny edges
    kernel = np.ones((3, 3), np.uint8)
    processed = cv.dilate(processed, kernel, iterations=1)

    contours, hierarchy = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #  check contours
    w,h,d = img.shape
    new_contours = []
    # for c in range(len(contours)):
        # check for image edges - "remove" if the detected edge connects with the image edge
        # if np.any(contours[c] == 0) or np.any(contours[c] == w-1):
            # continue
        # new_contours.append(contours[c])
    
    # find largest contour and assume that it's the correct one
    cntsSorted = sorted(new_contours, key=lambda x: cv.contourArea(x))
    # largest_contour = cntsSorted[-1]
    
    if len(contours) != 0:
        largest_contour = max(contours, key=cv.contourArea)
    
    cv.drawContours(contoured_image, [largest_contour], 0, (255, 0, 0), 1)

    mask = np.zeros((w, h)).astype(np.uint8)
    cv.fillPoly(mask, pts=[largest_contour], color=(255, 255, 255))

    return mask, largest_contour, contoured_image


def apply_mask(image, mask):

    return masked_image


def get_shape_factor(contour):
    # f = 4piA / P**2
    a = cv.contourArea(contour)
    p = cv.arcLength(contour, True)
    f = np.divide(4 * np.pi * a, p**2) 
    return f


data = pd.read_csv(paths["data"] + "/metadata.csv")
data["id_int"] = [i.split("_")[1] for i in data["isic_id"]]
data = data.sort_values(by=["id_int"], ascending=True).reset_index()
data = data.head(n_images)

data["image_raw"] = [cv.imread(path.join(paths["images"], i + ".JPG")) for i in data["isic_id"]]
data["dca"] = [get_dca(i) for i in data["image_raw"]]
data["dca_rm"] = [remove_DCA(image=data["dca"][i][0], mask=data["dca"][i][1], removal_method='inpaint_ns') for i in range(len(data["image_raw"]))]
data["image_proc"] = [process_img(i) for i in data["dca_rm"]]
# data["image_masks"], data["contours"], data["contoured_image"] = map(list,
#         zip(*[segment(i) for i in
#     data["image_proc"]]))
# data["image_masked"] = [cv.bitwise_and(data["image_proc"][i], data["image_proc"][i],
#     mask=data["image_masks"][i].astype(np.uint8)) for i in
#     range(len(data["image_proc"]))]
# shape_factor = [get_shape_factor(c) for c in data["contours"]]


[grabcut_segment(i) for i in data["image_proc"]]

# for i in range(n_images):
#     img = [
#             data["image_proc"][i], data["contoured_image"][i], data["image_masked"][i]
#     ]
#     labels = ["Raw", "contour", "mask"]
