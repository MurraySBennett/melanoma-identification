import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter


def show_images(img1, img2):
    # img1 = cv.cvtColor(img1, cv.COLOR_LAB2RGB)
    # img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    f, ax = plt.subplots(1, 2, figsize=(5, 5))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[0].axis('off')
    ax[1].axis('off')
    f.tight_layout()
    plt.show()


def get_mean(img):
    img_mean = np.mean(img)
    img_tmp = img.copy()
    img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2] = np.average(img, axis=(0, 1))
    return img_mean, img_tmp


def combine_imgmask(img_path, mask_path):
    """ read img, read mask, return combined """
    img_id = split(img_path, '/', 
    print(img_id)
    try:
        img = cv.imread(img_path)
        mask = cv.imread(mask_path, -1)
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        print(img.shape, mask.shape)
        if img.shape[:2] != mask.shape[:2]:
            # Resize or crop the mask to match the image size
            print("incompatible image and mask sizes")
            # mask = cv.resize(mask, img.shape[:2], interpolation=cv.INTER_NEAREST)
        masked = impose_mask(img, mask)
        return [masked, img_id]
    except Exception as e:
        print(f"Error processing {img_path} and {mask_path}: {e}")
        return None


def impose_mask(img, mask):
    """ slap mask onto image """
    masked = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))
    return masked


def palette_perc(k_cluster, show_image=False):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    cluster_labels = k_cluster.labels_
    counter = Counter(k_cluster.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = dict(sorted(perc.items()))

    # percentages of each colour - you'll need this for checking the proportion of colour in the image
    # print(perc)
    # print(k_cluster.cluster_centers_)

    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)
    if show_image:
        return palette
    else:
        return perc


# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234352#sec002
def col_var(clustered_colours, colour_pct):
    colour_list = ["black", "dark-brown", "light-brown", "red", "white", "blue-gray"]
    black = [0, 0, 0]
    p = 3
    T = round(minkowski(black, white, p) / 2)

    minkowski_colours = [colour for idx, colour in enumerate(clustered_colours) if colour_pct[idx] >= 0.05]
    minkowksi_identified = [minkowski_check(c, p, T) for c in minkowski_colours]

    # minkowski_white = [minkowski(white, ic, p) < T for ic in minkowski_colours]
    # minkowski_red = [minkowski(red, ic, p) < T for ic in minkowski_colours]
    # minkowski_blue_gray = [minkowski(blue_gray, ic, p) < T for ic in minkowski_colours]
    # minkowski_check = np.array([np.any(minkowski_red), np.any(minkowski_white), np.any(minkowski_blue_gray)])


    # sus_high = np.array([black[1], dark_brown[1], light_brown[1]])
    # sus_low = np.array([black[0], dark_brown[0], light_brown[0]])

    # above_low = np.all(clustered_colours[None] >= sus_low[:, None], axis=2)
    # below_high = np.all(clustered_colours[None] <= sus_high[:, None], axis=2)

    # within_range = np.logical_and(above_low, below_high)
    # check_ranged_colours = np.array(within_range.sum(axis=1) > 0.01)
    range_identified_colours = [id_colour(c) for c in clustered_colours]
    # check_colours = list(check_ranged_colours) + list(minkowski_check)
    # identified_colours = [colour_list[i] for i, c in enumerate(check_colours) if c == True]
    identified_colours = range_identified + minkowski_identified

    return len(identified_colours), identified_colours


def check_colour_range(colour, colour_range):
    """ check if the colour is with the range """
    lower, upper = color_range
    return all(lower[i] <= colour[i] <= upper[i] for i in range(3))


def id_colour(colour):
    """ ID the colour based on it's hsv value """
    black = [[0.06, 0.27, 0.10], [39.91, 30.23, 22.10]]
    dark_brown = [[14.32, 6.85, 6.96], [47.57, 27.14, 46.81]]
    light_brown = [[47.94, 11.89, 19.86], [71.65, 44.81, 64.78]]

    if check_colour_range(colour, black):
        return "black"
    elif check_colour_range(colour, dark_brown):
        return "dark-brown"
    elif check_colour_range(colour, light_brown):
        return "light-brown"


def check_minkowski(colour, p=3, T=0.5):
    white = [100, 0, 0]
    red = [54.29, 80.81, 69.89]
    blue_gray = [50.28, -30.14, -11.96]
    if minkowski_distance(white, colour, p) < T:
        return "white"
    elif minkowski_distance(red, colour, p) < T:
        return "red"
    elif minkowski_distance(red, colour, p) < T:
        return "blue_gray"


    # I am here -- I have a list of 3 values, 1s or 0s, that represent whether the three ranged colours are present. I
    # want to return those colours, but I'm not sure how to index the colour_list with the ranged_colours variable.
    # once you figure that, the next job is to determine whether the white, red, and blue-grey are less than 50
    # minkowskis of each single_colour, and the proportion of that colour is greater than 5% of the lesion.

    # upon re-reading, I think they want to check every pixel of the lesion for the ranged colours. If that colour
    # exists **at all**, even 1 pixel, then it is 'present' in the lesion. Hold the phone. THey do a k-means analysis
    # and convert the pixels to the average of the cluster then check those values (essentially what you do above).

    # https://stackoverflow.com/questions/67276643/how-to-find-all-pixel-values-from-a-certain-range
    # above_low = np.all(img[None] >= sus_low[:, None, None], axis=3)  # .sum(axis=(1, 2))
    # below_high = np.all(img[None] <= sus_high[:, None, None], axis=3)
    # within_range = above_low + below_high
    # gives num pixels within range
    # counts = within_range.sum(axis=(1, 2))

    # the above is useful for counting the proportion of pixels for each colour.
    # You need to know if the clustered colour is within the range. Which means
    # you don't need to survey the entire image, just the colour list.

    # suspicious colour if within ranges (black, light/dark brown) or if distance between pixel
    # and single colour (red, blue-gray, white) is below threshold T

def minkowski_distance(x, y, p):
    return np.power(np.abs(x-y), p).sum() ** (1/p)


def get_clusters(img, n_clusters):
    # if n_clusters is an array, we find and return the best clustering
    if isinstance(n_clusters, list):
        for k in range(n_clusters[0], n_clusters[1]):
            kmeans_model = KMeans(k)
            kmeans_model.fit(img.reshape(-1, 3))
            sse.append(kmeans_model.inertia_)
        knee = KneeLocator(
            range(n_clusters[0], n_clusters[1]), sse, curve="convex", direction="decreasing"
        )
        kmeans_model = KMeans(knee.elbow)
    else:
        kmeans_model = KMeans(n_clusters, n_init='auto')

    kmeans_model.fit(img.reshape(-1, 3))
    return kmeans_model


