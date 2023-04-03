from __future__ import division
import os
import cv2
import numpy as np
from time import time, perf_counter
import concurrent.futures
from tqdm import tqdm

# local imports
from minimise_dca import *


## Parallel Processing
## https://www.machinelearningplus.com/python/parallel-processing-python/

def get_files(path, n_files):
    f = []
    names = []
    roots = []
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".JPG") or name.endswith(".jpg"):
                names.append(name)
                f.append(os.path.join(root, name))
                roots.append(root)
            if n_files is not None and count == n_files:
                break
            count += 1
    return f, names


def process_img(file_name):
    size = (512, 384)

    save_path = os.path.join(os.getcwd(), "..", "resized")
    img_path = os.path.join(os.getcwd(),  "..", "ISIC-database")

    img = cv2.imread(img_path + "/" + file_name)
    
    # remove white border from images
    trim_pct = 1
    img = trim_border(img,trim_pct)   

    # minimise dark corners
    img = get_dca(img)[0]
    

    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    # if np.round(w / h, 2) != np.round(4.0/3.0, 2):
    #     h_delta = (h - int(np.round(w * 0.75))) // 2
    #     img = img[h_delta:h-h_delta, :]
    #     print(h_delta)

    if w > size[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if w < size[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    if save_path is not None:
        save_file = save_path +"/" + file_name
        cv2.imwrite(save_file, img)


if __name__ == '__main__':
    start = perf_counter()
    n_files = None  # set to None to run all files
    img_path = "/mnt/c/Users/qlm573/melanoma-identification/images/ISIC-database/"
    f, names = get_files(img_path, n_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() -1) as executor:
        futures = [executor.submit(process_img, i) for i in names]
    end = perf_counter()
    print(f'time elapsed: {perf_counter()-start}')

