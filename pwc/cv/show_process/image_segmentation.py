import os 
from os import path
import pandas as pd
import glob
from pprint import pprint
import concurrent.futures
import cv2 as cv
from time import perf_counter

from image_processes import process_img
from file_management import save_img, read_img

batch_size = (os.cpu_count()-1) * 10
n_images = None # set to None if running all images- maybe do this on the HPC
save_data = True

##### --> set paths and filenames
home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
paths = dict(
    home=home_path,
    images=path.join(home_path, "images", "resized"),
    masks=path.join(home_path, "images", "segmented", "masks"),
    segmented=path.join(home_path, "images", "segmented", "images"),
    data=path.join(home_path, "images", "metadata")
    )
image_paths = glob.glob(os.path.join(paths['images'], "*.JPG"))
image_paths = sorted(image_paths)
if n_images is not None:
    image_paths = image_paths[:n_images]


def main():
    # data = pd.read_csv(paths["data"] + "/metadata.csv")
    # data["id_int"] = [i.split("_")[1] for i in data["isic_id"]]
    # data = data.sort_values(by=["id_int"], ascending=True).reset_index()
    # if n_images is not None:
        # data = data.head(n_images)
    # img_labels = [path.join(paths['images'], i + '.png') for i in data['isic_id']]
    # save_labels  = [path.join(paths['masks'], i + '.png') for i in data['isic_id']]


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
            batch_mask_paths = [os.path.join(paths['masks'], os.path.splitext(os.path.basename(p))[0] + ".png") for p in batch_paths]
            io_executor.map(save_img, batch_masks, batch_mask_paths)
            print(perf_counter()-start)


if __name__ == '__main__':
    main()
