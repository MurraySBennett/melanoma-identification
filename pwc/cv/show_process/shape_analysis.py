import os
import glob
import concurrent.futures
import numpy as np
import logging
import cv2 as cv
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "cv_shape.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def show_images(images):
    f, ax = plt.subplots(1, len(images), figsize=(5, 5))
    for i, img in enumerate(images):
        img[192, :] = 1
        img[:, 256] = 1
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
    f = np.divide(4 * np.pi * a, p**2) 
    return f


def get_area(contour):
    return cv.contourArea(contour)


def get_asymmetry(mask):
    #https://link.springer.com/article/10.1007/s42452-019-0786-8#Sec2
    contour = get_contour(mask)
    ellipse = cv.fitEllipse(contour)
    major_axis = max(ellipse[1])
    major_angle = ellipse[2]

    M = cv.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    theta = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])

    tx = int(mask.shape[1] / 2) - cx
    ty = int(mask.shape[0] / 2) - cy

    # rotation_matrix = cv.getRotationMatrix2D((cx, cy), -theta * 180 / np.pi, 1)
    if major_angle < 90:
        rotation_angle = major_angle + 90
    else:
        rotation_angle = major_angle - 90

    rotation_matrix = cv.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rotation_angle, 1)
    rotated_mask = cv.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
    # centered_mask = cv.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))
    # transformed_mask = cv.warpAffine(centered_mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
    show_images([mask, rotated_mask])#centered_mask, transformed_mask])

    return transformed_mask


def get_border_regularity(contour):
    return


def aspect_ratio(contour):
    return


def AP_ratio(contour):
    a = cv.contourArea(contour)
    p = cv.arcLength(contour, True)
    return round(a / p, 2)


def read_mask(path):
    mask = cv.imread(path, -1)
    return mask 


def get_contour(mask):
    # ret, thresh = cv.threshold(mask, 127, 255, 0)
    contour, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contour[0]


batch_size = (os.cpu_count()-1) * 2**4
n_images = 3#None
save_data = False

home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')

paths = dict(
    home=home_path,
    images=os.path.join(home_path, "images", "resized"),
    masks=os.path.join(home_path, "images", "segmented", "masks"),
    segmented=os.path.join(home_path, "images", "segmented", "images"),
    data=os.path.join(home_path, "images", "metadata")
    )
mask_paths = glob.glob(os.path.join(paths['masks'], '*.png'))
mask_paths = sorted(mask_paths)

if n_images is not None:
    mask_paths = mask_paths[:n_images]

def main():
    with concurrent.futures.ThreadPoolExecutor() as io_exec:
        for i in range(0, len(mask_paths), batch_size):
            batch_mask_paths = mask_paths[i:i+batch_size]
            batch_masks = list(io_exec.map(read_mask, batch_mask_paths))
            with concurrent.futures.ProcessPoolExecutor() as cpu_executor:
                shape_features = list(cpu_executor.map(get_asymmetry, batch_masks))
    # for i in range(0, len(batch_masks)):
    #     show(batch_masks, 'test_label')
    #     show(shape_features, 'test_alignment')

if __name__ == '__main__':
    main()
