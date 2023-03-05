import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from collections import Counter
from kneed import KneeLocator
from scipy.spatial.distance import minkowski


def show_images(img1, img2):
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


def read_img(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    return img


def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = dict(sorted(perc.items()))

    print(perc)
    print(k_cluster.cluster_centers_)

    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)

    return palette


# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234352#sec002
def col_var(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    p = 3
    black = [0, 0, 0]
    white = [100, 0, 0]
    T = round(minkowski(black, white, p) / 2)
    colour_list = ["black", "dark-brown", "light-brown", "red", "white", "blue-gray"]

    black = [[0.06, 0.27, 0.10], [39.91, 30.23, 22.10]]
    dark_brown = [[14.32, 6.85, 6.96], [47.57, 27.14, 46.81]]
    light_brown = [[47.94, 11.89, 19.86], [71.65, 44.81, 64.78]]
    red = [54.29, 80.81, 69.89]
    blue_gray = [50.28, -30.14, -11.96]
    sus_high = np.array([black[1], dark_brown[1], light_brown[1]])
    sus_low = np.array([black[0], dark_brown[0], light_brown[0]])
    identified_colours_eg = np.array([[0, 0, 0],
                                      [59.263, 6.519, -1.826],
                                      [47.051, 9.465, 2.307],
                                      [67.222, 3.355, -4.661],
                                      [22.237, 0.779, -3.284],
                                      [40.018, 10.696, 4.107],
                                      [53.075, 8.206, -0.027]
                                      ])
    above_low = np.all(identified_colours_eg[None] >= sus_low[:, None], axis=2)
    below_high = np.all(identified_colours_eg[None] <= sus_high[:, None], axis=2)
    # returns 3 rows ( 1 for each of the sus- ranged colours )
    # with 7 items in the row corresponding to the identified colour
    within_range = np.logical_and(above_low, below_high)
    ranged_colours = within_range.sum(axis=1)

    # I am here -- I have a list of 3 values, 1s or 0s, that represent whether the three ranged colours are present. I
    # want to return those colours, but I'm not sure how to index the colour_list with the ranged_colours variable.
    # once you figure that, the next job is to determine whether the white, red, and blue-grey are less than 50
    # minkowskis of each single_colour, and the proportion of that colour is greater than 5% of the lesion.

    # upon re-reading, I think they want to check every pixel of the lesion for the ranged colours. If that colour
    # exists **at all**, even 1 pixel, then it is 'present' in the lesion. Hold the phone. THey do a k-means analysis
    # and convert the pixels to the average of the cluster then check those values (essentially what you do above).



    # https://stackoverflow.com/questions/67276643/how-to-find-all-pixel-values-from-a-certain-range
    above_low = np.all(img[None] >= sus_low[:, None, None], axis=3)  # .sum(axis=(1, 2))
    below_high = np.all(img[None] <= sus_high[:, None, None], axis=3)
    within_range = above_low + below_high
    # gives num pixels within range
    counts = within_range.sum(axis=(1, 2))

    # the above is useful for counting the proportion of pixels for each colour.
    # You need to know if the clustered colour is within the range. Which means
    # you don't need to survey the entire image, just the colour list.

    # suspicious colour if within ranges (black, light/dark brown) or if distance between pixel
    # and single colour (red, blue-gray, white) is below threshold T


def get_clusters(img, n_clusters):
    # if n_clusters is an array, we find and return the best clustering
    if len(n_clusters) > 1:
        sse = []
        for k in range(n_clusters[0], n_clusters[1]):
            clt = KMeans(k)
            clt.fit(img.reshape(-1, 3))
            sse.append(clt.inertia_)
        knee = KneeLocator(
            range(n_clusters[0], n_clusters[1]), sse, curve="convex", direction="decreasing"
        )
        clt = KMeans(knee.elbow)
    else:
        clt = KMeans(n_clusters)

    clt.fit(img.reshape(-1, 3))
    return clt


img_path = os.path.join(os.getcwd(), "..", "..", "images", "ISIC-database")
os.chdir(img_path)
counter = 0
n_images = 3
min_clusters = 2
max_clusters = 7  # get 7 clusters, because we want 6 colours, and expect pure black to exist in the background.
for dirname, _, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        sse = []
        n_clusters = min_clusters
        if filename.endswith(".JPG"):
            print(os.path.join(dirname, filename))
            image = read_img(filename)
            mean_value, mean_colour = get_mean(image)
            clt = get_clusters(image, n_clusters=[min_clusters, max_clusters])
            show_images(image, palette_perc(clt))

            counter += 1
            if counter >= n_images:
                break
