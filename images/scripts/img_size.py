from __future__ import division
import os
from pathlib import Path
import cv2
import numpy as np
from time import time, perf_counter
import concurrent.futures
from tqdm import tqdm

# local imports
from minimise_dca import *


here = Path(__file__).resolve()
img_path = here.parent / "ISIC-database"
save_path = here.parent / "resized"

def main():
    start = perf_counter()
    n_files = None  # set to None to run all files
    f, names = get_files(img_path, n_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() -1) as executor:
        futures = [executor.submit(process_img, i) for i in names]
    print(f'time elapsed: {perf_counter()-start}')


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
    size = (256, 192)
    img = cv2.imread(img_path / file_name)

    # remove white border from images
    trim_pct = 1
    img = trim_border(img,trim_pct)

    # minimise dark corners -- removing this (1/9/2023) because I see a couple of images that shouldn't have been cropped. Perhaps the originals resized is as complex as it should be.
    # img = get_dca(img)[0]

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
        # save_file = save_path +"/" + file_name
        cv2.imwrite(save_path / file_name, img)


if __name__ == '__main__':
    main()
