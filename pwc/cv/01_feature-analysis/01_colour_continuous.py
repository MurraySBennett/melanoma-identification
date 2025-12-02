import os
from os import path
import numpy as np
import pandas as pd
import cv2 as cv
import logging
import glob
from time import perf_counter
import concurrent.futures

import matplotlib.pyplot as plt

from ...config import (PATHS, FILES)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
# file_handler = logging.FileHandler(FILES['cv_colour'])
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# logger.info(f'id,rms,coeff_1,coeff_2,coeff_3,mom1_1,mom2_1,mom3_1,mom4_1,mom1_2,mom2_2,mom3_2,mom4_2,mom1_3,mom2_3,mom3_3,mom4_3')

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
    result = {'id': label}
    if img is None:
        # logger.info(f'{label}')
        keys = ['rms', 'coeff_1', 'coeff_2', 'coeff_3', 'mom1_1', 'mom2_1', 'mom3_1', 'mom4_1', 'mom1_2', 'mom2_2', 'mom3_2', 'mom4_2', 'mom1_3', 'mom2_3', 'mom3_3', 'mom4_3']
        for key in keys:
            result[key] = np.nan
    else:
        try:
            rms_val = rms_colour(img)
            coeff = get_coeff_var(img)
            moments_1 = moments(img, 0)
            moments_2 = moments(img, 1)
            moments_3 = moments(img, 2)
            
            result.update({
                'rms': rms_val, 
                'coeff_1': coeff[0], 'coeff_2': coeff[1], 'coeff_3': coeff[2],
                'mom1_1': moments_1[0], 'mom2_1': moments_1[1], 'mom3_1': moments_1[2], 'mom4_1': moments_1[3],
                'mom1_2': moments_2[0], 'mom2_2': moments_2[1], 'mom3_2': moments_2[2], 'mom4_2': moments_2[3],
                'mom1_3': moments_3[0], 'mom2_3': moments_3[1], 'mom3_3': moments_3[2], 'mom4_3': moments_3[3]
            })
            
        except Exception as e:
            logger.error(f'Error calculating metics for {label}: {e}')
            for key in result.keys():
                if key != 'id':
                    result[key] = np.nan
    return result


def get_lesion(img_path, mask_path):
    label = os.path.basename(img_path)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    mask = cv.imread(mask_path, -1)
    try:
        # masked = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))
        # mask_target = np.all(mask != [0, 0, 0], axis=-1)
        mask_target = mask != 0
        lesion = img[mask_target]
        return [lesion, label]
    except Exception as e:
        print(f"Error processing {label}: {e}")
        return [None, label]

batch_size = (os.cpu_count()-1) * 2**6
n_images = None

image_paths = glob.glob(path.join(PATHS['images'], '*.JPG'))
image_paths = sorted(image_paths)
if n_images is not None:
    image_paths = image_paths[:n_images]
mask_paths = glob.glob(path.join(PATHS['masks'], '*.png'))
mask_paths = sorted(mask_paths)


def main():
    all_features = []
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
                all_features.extend(continuous_colours)
            end = perf_counter() 
            print(f'{counter} / {len(mask_paths)}: {np.round(counter / len(mask_paths) * 100,2)}%, est time remaining: {np.round((end-start)/batch_size * (len(mask_paths) - counter),2)/60}m, total: {np.round(end-initialise,2)}s')

    df = pd.DataFrame(all_features)
    column_order = [
        'id', 'rms', 'coeff_1', 'coeff_2', 'coeff_3', 
        'mom1_1', 'mom2_1', 'mom3_1', 'mom4_1', 
        'mom1_2', 'mom2_2', 'mom3_2', 'mom4_2', 
        'mom1_3', 'mom2_3', 'mom3_3', 'mom4_3'
    ]
    df = df[column_order]
    df.to_csv( FILES['cv_colour'], sep=',', index=False, na_rep='NA')

if __name__ == '__main__':
    main()

