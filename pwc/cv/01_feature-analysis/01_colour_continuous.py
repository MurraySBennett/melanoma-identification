import os
from pathlib import Path
from os import path
import numpy as np
import pandas as pd
import cv2 as cv
import logging
import glob
from time import perf_counter
import concurrent.futures

import matplotlib
matplotlib.use('Agg')
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

def worker_pipeline(paths):
    img_path, mask_path = paths
    label = Path(img_path).name
    try:
        img = cv.imread(str(img_path))
        if img is None: return None
        img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        if mask is None: return None
        
        mask_target = mask > 0
        lesion_pixels = img[mask_target]
        if lesion_pixels.size == 0:
            return {'id': label, **{k: np.nan for k in COLUMN_KEYS}}
        results = {'id': label}
        
        stds = np.std(lesion_pixels, axis=0)
        means = np.mean(lesion_pixels, axis=0)
        
        results['rms'] = np.sqrt(np.mean(stds**2))
        
        # Channels 0, 1, 2 (L, a, b)
        # for i in range(3):
        #     ch = lesion_pixels[:, i].astype(np.float64)
        #     m = means[i]
        #     s = stds[i]
            
        #     # Coeff Var (with guard for zero mean)
        #     results[f'coeff_{i+1}'] = s / m if m != 0 else 0
            
        #     # Moments
        #     results[f'mom1_{i+1}'] = m
        #     results[f'mom2_{i+1}'] = s
        #     results[f'mom3_{i+1}'] = np.mean(ch**3)
        #     results[f'mom4_{i+1}'] = np.mean(ch**4)
            
        return results

    except Exception as e:
        print(f"Error processing {label}: {e}")
        return None

# Global key order for consistency
COLUMN_KEYS = [
    'rms', #'coeff_1', 'coeff_2', 'coeff_3', 
    # 'mom1_1', 'mom2_1', 'mom3_1', 'mom4_1', 
    # 'mom1_2', 'mom2_2', 'mom3_2', 'mom4_2', 
    # 'mom1_3', 'mom2_3', 'mom3_3', 'mom4_3'
]
        


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
    # Use Pathlib for robust cross-platform pathing
    img_dir = Path(PATHS['images'])
    mask_dir = Path(PATHS['masks'])
    
    # Sort both to ensure they align (assuming identical filenames)
    img_paths = sorted(list(img_dir.glob('*.JPG')))
    mask_paths = sorted(list(mask_dir.glob('*.png')))
    
    # Pair them up
    path_pairs = list(zip(img_paths, mask_paths))
    
    all_features = []
    
    # Parallel Processing (Single Executor)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use a sensible chunksize to reduce overhead
        results = list(executor.map(worker_pipeline, path_pairs, chunksize=20))
        
    all_features = [r for r in results if r is not None]
    
    # Create DataFrame and enforce order
    df = pd.DataFrame(all_features)
    df = df[['id'] + COLUMN_KEYS]
    
    # Save with full precision (round at save if preferred)
    df.to_csv(FILES['cv_colour'], index=False, na_rep='NA')
    print(f"Done. Processed {len(all_features)} lesions.")

if __name__ == '__main__':
    main()

