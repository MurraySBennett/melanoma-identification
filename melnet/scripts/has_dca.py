# reference https://github.com/mmu-dermatology-research/dark_corner_artifact_removal/blob/master/Modules/image_modifications.py

from __future__ import division
import os
import cv2
from pathlib import Path
import numpy as np
from time import time, perf_counter
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from pprint import pprint as pp


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


def has_dca(file_name, divisor=10):
    img_id = file_name.split('.')[0]

    # save_path = os.path.join(os.getcwd(), "..", "resized")
    img_path = os.path.join(os.path.expanduser("~"), "win_home", "melanoma-identification", "images", "resized")
    img = cv2.imread(os.path.join(img_path, file_name + ".jpg"))

    total_px = img.shape[0] * img.shape[1]
    mask = np.ones(img.shape)
    mask[img <= 1] = 0
    mask[img > 1] = 255

    black_pixels = np.sum(mask == 0)
    white_pixels = np.sum(mask == 1)

    benchmark = total_px/divisor
    if black_pixels >= benchmark:
        badbadnotgood = {'id':img_id, 'prop': np.round(black_pixels / total_px, 3), 'reason': 'black'}
    if white_pixels >= benchmark:
        badbadnotgood = {'id':img_id, 'prop': np.round(white_pixels / total_px, 3), 'reason': 'white'}
        return badbadnotgood
    else:
        return
   

if __name__ == '__main__':
    start = perf_counter()
    n_files = None  # set to None to run all files
    
    base_dir = Path(os.getcwd()).parent.parent
    img_path = base_dir / "images" / "resized"
    data_path= base_dir / "data"
    data = pd.read_csv(data_path / "btl_cv_data_revised.csv")
    
    # img_path = os.path.join(os.path.expanduser("~"), "win_home", "melanoma-identification", "images", "resized")

    # scripts_path = os.path.join(os.path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melnet", "scripts")
    # data_path = os.path.join(os.path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melnet", "data")

    # data = pd.read_csv(os.path.join(data_path, "btl-cv-data.csv"))

    # print(f"n dca images = {data.shape}")
    # pp(data.head())

    # f, names = get_files(img_path, n_files)
    dca_ids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() -1) as executor:
        futures = [executor.submit(has_dca, i) for i in data["id"]]
        for future in concurrent.futures.as_completed(futures):
            img_id = future.result()
            if img_id is not None:
                dca_ids.append(img_id)

    dca_ids = pd.DataFrame(dca_ids)
    dca_ids.to_csv(os.path.join(data_path, "dca_images.txt"), index=False)

    end = perf_counter()
    print(f'time elapsed: {perf_counter()-start}')
    print(f"n dca images = {dca_ids.shape[0]}")

