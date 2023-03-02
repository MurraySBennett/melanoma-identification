import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from collections import Counter
from kneed import KneeLocator


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


img_path = os.path.join(os.getcwd(), "..", "..", "images", "ISIC-database")
os.chdir(img_path)
counter = 0
n_images = 3
min_clusters = 2
max_clusters = 6
for dirname, _, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        sse = []
        n_clusters = min_clusters
        if filename.endswith(".JPG"):
            print(os.path.join(dirname, filename))
            image = read_img(filename)
            mean_value, mean_colour = get_mean(image)
            print(mean_value)
            for k in range(min_clusters, max_clusters):
                clt = KMeans(n_clusters)
                clt.fit(image.reshape(-1, 3))
                sse.append(clt.inertia_)
            knee = KneeLocator(
                range(min_clusters, max_clusters), sse, curve="convex", direction="decreasing"
            )

            print(knee.elbow)
            clt = KMeans(n_clusters=knee.elbow)
            clt.fit(image.reshape(-1, 3))
            show_images(image, palette_perc(clt))

            counter += 1
            if counter >= n_images:
                break
