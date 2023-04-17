from time import perf_counter
import os
import logging
import glob
import cv2 as cv
import numpy as np
import concurrent.futures 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler('mask_pct.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# logger.addHandler(stream_handler)

logger.info('isic_id, segment_size')
n_images = 2
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
batch_size = os.cpu_count()-1

if n_images is not None:
    mask_paths = mask_paths[:n_images]


def get_segment_pct(img):
    label = img.split('/')[-1]
    img = cv.imread(img,-1)
    img = img // 255
    size = img.shape[0] * img.shape[1]
    white_sum = np.sum(img)
    black_sum = np.sum(img==0)
    white_pct = np.round(white_sum / size * 100)
    logger.info(f'{label}, {white_pct}')
    

def main():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(mask_paths), batch_size):
            batch_paths = mask_paths[i:i+batch_size]
            executor.map(get_segment_pct, batch_paths)
           

if __name__ == '__main__':
    main()
