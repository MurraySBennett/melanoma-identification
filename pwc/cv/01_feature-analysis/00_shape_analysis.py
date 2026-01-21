import os
from pathlib import Path
import glob
import concurrent.futures
import numpy as np
import pandas as pd
import logging
import cv2 as cv
import matplotlib
matplotlib.use('Agg') # force non-interactice backend
import matplotlib.pyplot as plt
from time import perf_counter

from ...config import (FILES, PATHS)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

# file_handler = logging.FileHandler(FILES['cv_shape'], 'w') # overwrite/ new file on init
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def show_images(images):
    f, ax = plt.subplots(1, len(images), figsize=(len(images)*1.5, 3))
    h, w = images[0].shape
    for i, img in enumerate(images):
        img[h//2, :] = 0
        img[:, w//2] = 0
        ax[i].imshow(img)
        ax[i].axis('off')
        f.tight_layout()
    plt.show()


def show(img, label):
    cv.imshow(f'{label}', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def get_shape_factor(contour):
    """f = 4piA / P**2"""
    # low scores = irregular
    # "circularity index" -- Features for Melanoma Lesions Characterization in Computer Vision Systems
    a = cv.contourArea(contour)
    p = cv.arcLength(contour, True)
    if p == 0:
        return 0.0
    f = np.divide(4 * np.pi * a, p**2) 
    return np.round(f, 4)


def get_area(contour):
    return cv.contourArea(contour)


def center_segment(mask, contour):
    M = cv.moments(contour)
    # if M['m00'] == 0:
    #     return mask
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # theta = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
    tx = int(mask.shape[1] / 2) - cx
    ty = int(mask.shape[0] / 2) - cy
    translation_matrix = np.float32([[1, 0, tx],[0, 1, ty]])
    return cv.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))


def align_major_axis(mask, contour):
    # ellipse = cv.fitEllipse(contour)
    # major_axis = max(ellipse[1])
    # major_angle = ellipse[2]
    # rotation_angle = major_angle + 90 if major_angle < 90 else major_angle - 90
    # rotation_matrix = cv.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rotation_angle, 1)
    # rotated_mask = cv.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
    # return rotated_mask

    M = cv.moments(contour)
    if abs(M['mu20'] - M['mu02']) > 1e-5:
        theta = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
        rotation_angle  = np.degrees(theta)
    else:
        rotation_angle = 0.0
    center = (mask.shape[1] / 2, mask.shape[0] / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1)
    return cv.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))


def rotate_mask(mask, rotation_angle=90):
    rotation_angle = 90
    rotation_matrix = cv.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rotation_angle, 1)
    rotated_mask = cv.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
    return rotated_mask


def flip_mask_h(mask, section = 'top'):
    h, w = mask.shape
    midpoint = h // 2
    flipped_mask = mask.copy()
    if section == 'top':
        mirror = cv.flip(mask[:midpoint, :], 0)
        flipped_mask[midpoint:, :] = mirror
    elif section == 'bottom':
        mirror = cv.flip(mask[midpoint:, :], 0)
        flipped_mask[:midpoint, :] = mirror
    return flipped_mask


def combine_halves(mask, axis='horz'):
    h, w = mask.shape
    midpoint = h // 2
    combined = mask.copy()
    if axis == 'horz':
        mirror = cv.flip(mask[midpoint:, :],0)
        combined[:midpoint,:] = (mask[:midpoint,:] + mirror) % 2
        combined = flip_mask_h(combined, section='top')
    return combined



def get_asymmetry(mask, contour):
    #https://link.springer.com/article/10.1007/s42452-019-0786-8#Sec2
    centered_mask = center_segment(mask, contour)
    rotated_mask = align_major_axis(centered_mask, contour)

    # L = rotated_mask
    # Lx = cv.flip(L, 0)
    # dLx= (L + Lx) % 2
    # Ly = cv.flip(L, 1)
    # dLy= (L + Ly) % 2
    # x_symmetry = np.round(np.divide(np.sum(dLx), np.sum(Lx)), 4)
    # y_symmetry = np.round(np.divide(np.sum(dLy), np.sum(Ly)), 4)
    # return [x_symmetry, y_symmetry]    
    
    L = (rotated_mask > 0).astype(np.uint8)
    Lx = cv.flip(L, 0)
    dLx = cv.bitwise_xor(L, Lx)
    Ly = cv.flip(L, 1)
    dLy = cv.bitwise_xor(L, Ly)

    sum_L = np.sum(L)
    if sum_L == 0:
        return [np.nan, np.nan]
    x_asym = np.round(np.sum(dLx) / sum_L, 4)
    y_asym = np.round(np.sum(dLy) / sum_L, 4)
     
    return [x_asym, y_asym]


def AP_ratio(contour):
    a = cv.contourArea(contour)
    p = cv.arcLength(contour, True)
    return round(a / p, 4)


def measure_shape(mask):
    label = mask[1]
    mask = mask[0]
    contour = get_contour(mask)
    try:
        x_sym, y_sym = get_asymmetry(mask, contour)
        compact_factor = get_shape_factor(contour)
        ap_ratio = AP_ratio(contour)
        # logger.info(f'{label},{x_sym},{y_sym},{compact_factor},{ap_ratio}')
        # shape_values = {'id':label, 'x_sym':x_sym, 'y_sym':y_sym, 'compact':compact_factor,'ap_ratio': ap_ratio}
        return {
            'id': label, 
            'x_sym': x_sym, 
            'y_sym': y_sym, 
            'compact': compact_factor, 
            'ap_ratio': ap_ratio
        }
    
    except Exception as e:
        logger.error(f'Failed to process {label}: {e}')
        return {
            'id': label, 
            'x_sym': np.nan, 
            'y_sym': np.nan, 
            'compact': np.nan, 
            'ap_ratio': np.nan
        }


def read_mask(path):
    mask = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if mask is None:
        return [None, Path(path).name]
    _, mask_bin = cv.threshold(mask, 127, 1, cv.THRESH_BINARY)
    return [mask_bin, Path(path).name] 


def get_contour(mask):
    contour, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contour[0]


def worker_pipeline(mask_path):
    """ read, process, return results. """
    try:
        mask, mask_id = read_mask(mask_path)
        if mask is None:
            return None
        contour = get_contour(mask)
        x_sym, y_sym = get_asymmetry(mask, contour)
        compact_factor = get_shape_factor(contour)
        ap_ratio = AP_ratio(contour)
        return {
            'id': mask_id,
            'x_sym': x_sym,
            'y_sym': y_sym,
            'compact': compact_factor,
            'ap_ratio': ap_ratio
        }
    except Exception as e:
        print(f"Failed to process {os.path.basename(mask_path)}: {e}")


def main():
    batch_size = 128 #(os.cpu_count()-1) * 2**6
    n_images = None

    mask_dir = Path(PATHS['masks'])
    mask_paths = sorted(list(mask_dir.glob('*.png')))

    if n_images is not None:
        mask_paths = mask_paths[:n_images]
        counter = 0
    
    # logger.info('id,x_sym,y_sym,compact,ap_ratio')
    all_features = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_pipeline, mask_paths, chunksize=10))
    all_features = [r for r in results if r is not None]
    
    df = pd.DataFrame(all_features)
    df.to_csv( FILES['cv_shape'], index=False, na_rep='NA')
    print(f"Shape featurs saved to: {FILES['cv_shape']} ")

    # for i in range(0, len(batch_masks)):
    #     show(batch_masks, 'test_label')
    #     show(shape_features, 'test_alignment')

if __name__ == '__main__':
    main()

