import os
import glob
import concurrent.futures
from time import perf_counter
from os import path

from colour_categorisation import *
from shape_analysis import *


batch_size = (os.cpu_count()-1)
n_images = 10# None
save_data = False
min_clusters = 2
max_clusters = 7  # get 7 clusters, because we want 6 colours, and expect pure black to exist in the background.

home_path =  '/mnt/c/Users/qlm573/melanoma-identification/'
paths = dict(
    home=home_path,
    images=path.join(home_path, "images", "resized"),
    masks=path.join(home_path, "images", "segmented", "masks"),
    segmented=path.join(home_path, "images", "segmented", "images"),
    data=path.join(home_path, "images", "metadata")
    )
image_paths = glob.glob(os.path.join(paths['images'], '*.JPG'))
image_paths = sorted(image_paths)
if n_images is not None:
    image_paths = image_paths[:n_images]
mask_paths = glob.glob(os.path.join(paths['masks'], '*.png'))
mask_paths = sorted(mask_paths)


def main():
    counter = 0
    with concurrent.futures.ThreadPoolExecutor() as io_exec:
        for i in range(0, len(image_paths), batch_size):
            start = perf_counter()
            batch_img_paths = image_paths[i:i+batch_size]
            batch_mask_paths = mask_paths[i:i+batch_size]
            print(f'Batch {i//batch_size+1}: {len(batch_img_paths)} images and masks')
            batch_images = list(io_exec.map(combine_imgmask, batch_img_paths, batch_mask_paths))
            counter += len(batch_images)
    
            # Use ProcessPoolExecutor to perform image processing on each batch of images
            # with concurrent.futures.ProcessPoolExecutor() as cpu_executor:
            #     feature_values = list(cpu_executor.map(get_features, batch_images))

            # Use ThreadPoolExecutor to save the segmented masks in parallel
            # batch_mask_paths = [os.path.join(paths['masks'], os.path.splitext(os.path.basename(p))[0] + ".png") for p in batch_paths]
            # io_executor.map(save_img, batch_masks, batch_mask_paths)

            end = perf_counter()
            print(f'Processed {counter} images in {end-start:.2f} seconds')
            # sse = []
            # n_clusters = min_clusters
            # if filename.endswith(".JPG"):
            #     # print(os.path.join(dirname, filename))
            #     image = read_img(filename)
            #     mean_value, mean_colour = get_mean(image)
            #     # clt = get_clusters(image, n_clusters=[min_clusters, max_clusters])
            #     clt = get_clusters(image, n_clusters=max_clusters)

            #     cluster_pct = palette_perc(clt)
            #     n_colours, colours = col_var(clt.cluster_centers_, cluster_pct)
            #     print(colours)
            #     show_images(image, palette_perc(clt, show_image=True))

            #     counter += 1
            #     if counter >= n_images:
            #         break
    show_images(batch_images[0], batch_images[1])    

if __name__ == '__main__':
    main()

