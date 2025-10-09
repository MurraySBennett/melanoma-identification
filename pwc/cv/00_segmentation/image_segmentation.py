import os
from pathlib import Path
import pandas as pd
import glob
from pprint import pprint
import concurrent.futures
import cv2 as cv
from time import perf_counter

from ...config import PATHS

from .image_processes import process_img
from .file_management import save_img, read_img

batch_size = (os.cpu_count()-1) * 10
n_images = None # set to None if running all images- maybe do this on the HPC
save_data = True

image_paths = sorted(list(PATHS["images"].glob("*.JPG")))
if n_images is not None:
    image_paths = image_paths[:n_images]


def main():
    # Use ThreadPoolExecutor to read images in batches
    with concurrent.futures.ThreadPoolExecutor() as io_executor:
        for i in range(0, len(image_paths), batch_size):
            start=perf_counter()
            batch_paths = image_paths[i:i+batch_size]
            batch_images = list(io_executor.map(read_img, batch_paths))

            # Use ProcessPoolExecutor to perform image processing on each batch of images
            with concurrent.futures.ProcessPoolExecutor() as cpu_executor:
                batch_masks = list(cpu_executor.map(process_img, batch_images))

            # Use ThreadPoolExecutor to save the segmented masks in parallel
            batch_mask_paths = [
                PATHS["masks"] / f"{p.stem}.png" for p in batch_paths
            ]
            io_executor.map(save_img, batch_masks, batch_mask_paths)
            print(perf_counter()-start)


if __name__ == '__main__':
    main()
