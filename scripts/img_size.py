from __future__ import division
import os
import cv2
import numpy as np
from time import time, perf_counter
import concurrent.futures
from tqdm import tqdm

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


def img_process_args(args):
    return process_img(*args)


def process_img(file_name):
    size = (512, 384)  # 256,192
    save_path = os.path.join(os.getcwd(), "..", "images", "resized")
    img_path = os.path.join(os.getcwd(),  "..", "images", "ISIC-database")

    img = cv2.imread(img_path + "\\" + file_name)
    h, w = img.shape[:2]

    if h > w:
        # set width as longest side
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    if np.round(w / h, 2) != np.round(4.0/3.0, 2):
        # set aspect to 4:3 by setting the height to be 0.75*width
        h = int(np.round(w * 0.75))
        img = img[0:h, :]  # img[rows,columns]

    # resize images for memory -- I think they'll all be resized down to 256,192
    # for the experiment and the image analysis.
    if w > size[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if w < size[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    if save_path is not None:
        save_file = save_path +"\\" + file_name
        cv2.imwrite(save_file, img)


if __name__ == '__main__':
    # This takes about half an hour to do all 71, 670 images.
    start = perf_counter()
    n_files = None  # set to None to run all files.
    img_path = os.path.join(os.getcwd(), "..", "images")
    f, names = get_files(os.path.join(img_path, "ISIC-database"), n_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() -1) as executor:
        futures = [executor.submit(process_img, i) for i in names]
    end = perf_counter()
    print(f"total runtime for all images = {end - start}s")
