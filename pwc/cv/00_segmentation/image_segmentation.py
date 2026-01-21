import os
from pathlib import Path
import pandas as pd
import glob
from pprint import pprint
# import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import cv2 as cv
from time import perf_counter

from ...config import PATHS

from .image_processes import process_img
from .file_management import save_img, read_img

BATCH_SIZE = 50 #(os.cpu_count()-1) * 10
n_images = None # set to None if running all images- maybe do this on the HPC
save_data = True


def worker_task(image_path):
    """Internal worker: handle reading, procesing, and saving in one go.
    Args:
        image_path (str): image path..
    """
    try:
        img = read_img(image_path)
        mask = process_img(img)
        save_path = PATHS["masks"] / f"{image_path.stem}.png"
        save_img(mask, save_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return False


def main():
    image_paths = sorted([
        p for p in PATHS["images"].iterdir()
        if p.suffix.lower() in ['.jpg', '.jpeg']
    ])

    if n_images is not None:
        image_paths = image_paths[:n_images]

    # Use ThreadPoolExecutor to read images in batches
    with ProcessPoolExecutor() as executor:
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i:i+BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1}/{len(image_paths)//BATCH_SIZE}...")
            results = list(executor.map(worker_task, batch))


if __name__ == '__main__':
    main()
