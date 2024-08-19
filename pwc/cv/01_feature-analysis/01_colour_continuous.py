import os
from os import path
import numpy as np
import cv2 as cv
import logging
import glob
from time import perf_counter
import concurrent.futures

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv_colour.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f'id    rms coeff_1 coeff_2 coeff_3 mom1_1  mom2_1  mom3_1  mom4_1  mom1_2  mom2_2  mom3_2  mom4_2  mom1_3  mom2_3  mom3_3  mom4_3')

def rms(values):
    squared = [x ** 2 for x in values]
    return np.round(np.sqrt(np.sum(squared) / len(values)), 3)


def rms_colour(img):
    channel_1 = np.round(np.std(img[:,0]), 3)
    channel_2 = np.round(np.std(img[:,1]), 3)
    channel_3 = np.round(np.std(img[:,2]), 3)
    return rms([channel_1, channel_2, channel_3])


def moments(img, channel):
    first = np.round(np.mean(img[:, channel]), 3)
    second= np.round(np.std(img[:, channel]), 3)
    third = np.round(np.mean(img[:, channel]**3), 3) # estimate skew by dividing this value by the standard deviation
    fourth= np.round(np.mean(img[:, channel]**4), 3) # estimate kurtosis by dividing this value by sd**4
    return [first, second, third, fourth]


def get_hist(img):
    hist_1 = cv.calcHist([img], [0], None, [256], [0, 256]) 
    hist_2 = cv.calcHist([img], [1], None, [256], [0, 256]) 
    hist_3 = cv.calcHist([img], [2], None, [256], [0, 256]) 
    return [hist_1, hist_2, hist_3]


def plot_colour_hist(hists):
    plt.figure(figsize=(12,6))
    plt.subplot(131)
    plt.plot(hists[0], color='black')
    plt.title('Channel 1')

    plt.subplot(132)
    plt.plot(hists[1], color='green')
    plt.title('Channel 2')

    plt.subplot(133)
    plt.plot(hists[2], color='blue')
    plt.title('Channel 3')

    plt.tight_layout()
    plt.show()


# coefficient of variation
def get_coeff_var(img):
    channel_1 = np.round(np.std(img[:,0]) / np.mean(img[:,0]),3)
    channel_2 = np.round(np.std(img[:,1]) / np.mean(img[:,1]),3)
    channel_3 = np.round(np.std(img[:,2]) / np.mean(img[:,2]), 3)
    return [channel_1, channel_2, channel_3]


def get_metrics(img):
    label = img[1]
    img = img[0]
    if img is None:
        logger.info(f'{label}')
    else:
        rms = rms_colour(img)
        coeff = get_coeff_var(img)
        moments_1 = moments(img, 0) # channel 0
        moments_2 = moments(img, 1) # channel 1
        moments_3 = moments(img, 2) # channel 2
        # return [rms] + coeff + moments_1 + moments_2 + moments_3
        logger.info(f'{label}   {rms}   {coeff[0]}  {coeff[1]}  {coeff[2]}  {moments_1[0]}  {moments_1[1]}  {moments_1[2]}  {moments_1[3]}  {moments_2[0]}  {moments_2[1]}  {moments_2[2]}  {moments_2[3]}  {moments_3[0]}  {moments_3[1]}  {moments_3[2]}  {moments_3[3]}')


def get_lesion(img_path, mask_path):
    label = img_path.split('/')[-1]
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    mask = cv.imread(mask_path, -1)
    try:
        # masked = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))
        mask_target = np.all(mask != [0, 0, 0], axis=-1)
        lesion = img[mask_target]
        return [lesion, label]
    except Exception as e:
        print(f"Error processing {label}: {e}")
        return [None, label]

home_path = path.join(path.expanduser('~'), 'win_home', 'melanoma-identification')
batch_size = (os.cpu_count()-1) * 2**6
n_images = None

paths = dict(
    home=home_path,
    images=path.join(home_path, "images", "resized"),
    masks=path.join(home_path, "images", "segmented", "masks"),
    segmented=path.join(home_path, "images", "segmented", "images"),
    data=path.join(home_path, "images", "metadata")
    )
image_paths = glob.glob(path.join(paths['images'], '*.JPG'))
image_paths = sorted(image_paths)
if n_images is not None:
    image_paths = image_paths[:n_images]
mask_paths = glob.glob(path.join(paths['masks'], '*.png'))
mask_paths = sorted(mask_paths)


def main():
    counter = 0
    with concurrent.futures.ThreadPoolExecutor() as io_exec:
        initialise = perf_counter()
        for i in range(0, len(image_paths), batch_size):
            start = perf_counter()
            batch_img_paths = image_paths[i:i+batch_size]
            batch_mask_paths = mask_paths[i:i+batch_size]
            batch_images = list(io_exec.map(get_lesion, batch_img_paths, batch_mask_paths))
            counter += len(batch_images)

            with concurrent.futures.ProcessPoolExecutor() as cpu_executor:
                continuous_colours = list(cpu_executor.map(get_metrics, batch_images))
            end = perf_counter() 
            print(f'{counter} / {len(mask_paths)}: {np.round(counter / len(mask_paths) * 100,2)}%, est time remaining: {np.round((end-start)/batch_size * (len(mask_paths) - counter),2)/60}m, total: {np.round(end-initialise,2)}s')

if __name__ == '__main__':
    main()

# for i in range(0, len(image_paths), batch_size):
#     img_id = image_paths[i].split('/')[-1]
#     mask_id = mask_paths[i]

#     img = cv.imread(image_paths[i])
#     img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
#     mask = cv.imread(mask_paths[i], -1)

#     masked = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))
#     masked_lab = cv.bitwise_and(img_lab, img_lab, mask=mask.astype(np.uint8))

#     masked = combine_imgmask(image_paths[i], mask_paths[i])
#     # img = combine_imgmask(image_paths[i], mask_paths[i])
#     # mask_target = np.any(mask != [0, 0, 0], axis=-1)
#     mask_target = np.all(mask != [0, 0, 0], axis=-1)
#     lesion = img[mask_target]
#     lesion_lab = img_lab[mask_target]

#     # hists = get_hist(lesion)
#     # plot_colour_hist(hists)

#     # root mean square
#     rms_LAB = rms_colour(lesion_lab)
#     rms_RGB = rms_colour(lesion)

#     # coefficient of variation (sd/mu)
#     LAB_coeff = get_coeff_var(lesion_lab)
#     RGB_coeff = get_coeff_var(lesion)

#     # moments
#     # LAB_moments = moments(lesion_lab, 0)
#     # RGB_moments = moments(lesion, 0)
#     # LAB_moments = moments(lesion_lab, 1)
#     # RGB_moments = moments(lesion, 1)
#     # LAB_moments = moments(lesion_lab, 2)
#     # RGB_moments = moments(lesion, 2)

#     # print(rms_LAB, rms_RGB)
#     # show(masked, img_id)


