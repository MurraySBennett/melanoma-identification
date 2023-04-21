# import os
import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

file_handler = logging.FileHandler('colours.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# logger.addHandler(stream_handler)

# add headers
logger.info(f'isic_id, identified, mask_pct, pct_dict, colour_dict') 


def show(img, label):
    cv.imshow(f'{label}', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def combine_imgmask(img_path, mask_path):
    """ read img, read mask, return combined """
    img_id = img_path.split('/')[-1]
    try:
        img = cv.imread(img_path)
        mask = cv.imread(mask_path, -1)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if img.shape[:2] != mask.shape[:2]:
            # Resize or crop the mask to match the image size
            # mask = cv.resize(mask, img.shape[:2], interpolation=cv.INTER_NEAREST)
            print("mask was resized")

        masked = impose_mask(img, mask)
        return [masked, img_id]
    except Exception as e:
        logger.exception(f"Error processing {img_path} and {mask_path}: {e}")
        return None


def show_images(img1, img2):
    try:
        f, ax = plt.subplots(1, 2, figsize=(5, 5))
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        ax[0].axis('off')
        ax[1].axis('off')
        f.tight_layout()
        plt.show()
    except:
        logger.exception('Tried to show your images')


def get_mean(img):
    img_mean = np.mean(img)
    img_tmp = img.copy()
    img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2] = np.average(img, axis=(0, 1))
    return img_mean, img_tmp


# def combine_imgmask(img_path, mask_path):
#     """ read img, read mask, return combined """
#     try:
#         img = cv.imread(img_path)
#         mask = cv.imread(mask_path, -1)
#         img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
#         print(img.shape, mask.shape)
#         if img.shape[:2] != mask.shape[:2]:
#             # Resize or crop the mask to match the image size
#             print("incompatible image and mask sizes")
#             # mask = cv.resize(mask, img.shape[:2], interpolation=cv.INTER_NEAREST)
#         masked = impose_mask(img, mask)
#         return [masked, img_id]
#     except Exception as e:
#         print(f"Error processing {img_path} and {mask_path}: {e}")
#         return None

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
        perc[i] = np.round(counter[i] / n_pixels, 4)
    perc = dict(sorted(perc.items()))

    if show_image:
        step = 0
        for idx, centers in enumerate(k_cluster.cluster_centers_):
            palette[:, step:int(step + perc[idx] * width + 1), :] = centers
            step += int(perc[idx] * width + 1)
        return palette
    else:
        return perc


# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234352#sec002
def get_colours(img):
    label = img[1]
    img = img[0].astype('float32') / 255 
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    clusters = get_clusters(img)
    colour_pct = palette_perc(clusters)
    
    clustered_colours = np.round(clusters.cluster_centers_,2)
    
    # logger.info(f'{clustered_colours}')

    black_index = np.where(np.all(np.round(clustered_colours) == [0, 0, 0], axis=1))[0][0]
    clustered_colours = np.delete(clustered_colours, black_index,axis=0)
    mask_pct = 1 - colour_pct[black_index]
    del colour_pct[black_index]
    total_pct = sum(colour_pct.values()) 
    for key in colour_pct:
        colour_pct[key] = np.round((colour_pct[key] / total_pct) * 100)
    new_pct = {}
    save_clr={}
    new_key = 0
    for old_key in sorted(colour_pct.keys()):
        new_pct[new_key] = colour_pct[old_key]
        # save_clr[new_key] = clustered_colours[new_key].tolist()
        save_clr[new_key] = list(np.round(clustered_colours[new_key],2))
        new_key += 1
    colour_pct = new_pct
    del new_pct
    # colour_list = ["black", "dark-brown", "light-brown", "red", "white", "blue-gray"]
    # black = [0, 0, 0]
    # white = [100, 0, 0]
    # T = round(minkowski(black, white, p) / 2)
    # only use colours that constitute more than 5% of the image.
    minkowski_colours = [colour for idx, colour in enumerate(clustered_colours) if colour_pct[idx] >= 5]
    minkowski_identified = [check_minkowski(c, p=3, T=50) for c in minkowski_colours]
    
    range_identified_colours = [id_colour(c) for c in clustered_colours]
    # logger.info(range_identified_colours)
    identified_colours = range_identified_colours + minkowski_identified
    identified_colours = list(set([c for sublists in identified_colours for c in sublists if c is not None]))
    # identified_colours = [c for c in identified_colours if c is not None]
    logger.info(f'{label},{identified_colours}, {mask_pct}, {colour_pct}, {save_clr}')
    return identified_colours


def check_colour_range(colour, colour_range):
    """ check if the colour is with the range """
    lower, upper = colour_range
    return all(lower[i] <= colour[i] <= upper[i] for i in range(3))


def id_colour(colour):
    """ ID the colour based on it's hsv value """
    black = [[0.06, 0.27, 0.10], [39.91, 30.23, 22.10]]
    dark_brown = [[14.32, 6.85, 6.96], [47.57, 27.14, 46.81]]
    light_brown = [[47.94, 11.89, 19.86], [71.65, 44.81, 64.78]]
    colours = [None]
    if check_colour_range(colour, black):
        colours.append("black")
    if check_colour_range(colour, dark_brown):
        colours.append("dark-brown")
    if check_colour_range(colour, light_brown):
        colours.append("light-brown")
    return colours

def check_minkowski(colour, p=3, T=50):
    black = [0, 0, 0]
    white = [100, 0, 0]
    red = [54.29, 80.81, 69.89]
    blue_gray = [50.28, -30.14, -11.96]
    colours = [None]
    T = np.round(minkowski_distance(black, white, p) / 2)
    # minkowski_white = minkowski_distance(white, colour, p)
    # mink_red = minkowski_distance(red, colour, p)
    # mink_bg = minkowski_distance(blue_gray, colour, p)
    # logger.info(f'colour: {colour}, threshold: {t}, white: {minkowski_white}, red: {mink_red}, blue-gray: {mink_bg}')
    if minkowski_distance(white, colour, p) < T:
        colours.append("white")
    if minkowski_distance(red, colour, p) < T:
        colours.append("red")
    if minkowski_distance(blue_gray, colour, p) < T:
        colours.append("blue-gray")
    return colours


def minkowski_distance(x, y, p):
    """ return m-d for lab colour space values """
    lab_color1 = np.array(x)
    lab_color2 = np.array(y)
    diff = lab_color1 - lab_color2
    diff_abs = np.abs(diff)
    minkowski_dist = np.sum(diff_abs**p)**(1/p)
    return minkowski_dist
    # return np.power(np.abs(x-y), p).sum() ** (1/p)


def get_clusters(img, n_clusters=7):
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

img_path = os.path.join(os.getcwd(), "..", "..", "images", "ISIC-database")
os.chdir(img_path)
counter = 0
n_images = 1
min_clusters = 2
max_clusters = 7  # get 7 clusters, because we want 6 colours, and expect pure black to exist in the background.
for dirname, _, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        sse = []
        n_clusters = min_clusters
        if filename.endswith(".JPG"):
            # print(os.path.join(dirname, filename))
            image = read_img(filename)
            mean_value, mean_colour = get_mean(image)
            # clt = get_clusters(image, n_clusters=[min_clusters, max_clusters])
            clt = get_clusters(image, n_clusters=max_clusters)

            cluster_pct = palette_perc(clt)
            n_colours, colours = col_var(clt.cluster_centers_, cluster_pct)
            print(colours)
            show_images(image, palette_perc(clt, show_image=True))

            counter += 1
            if counter >= n_images:
                break
