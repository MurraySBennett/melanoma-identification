from time import perf_counter
import os
import glob
import concurrent.futures
from os import path
import logging
import numpy as np


from colour_categorisation import combine_imgmask, show, get_colours

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler('logging.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


batch_size = (os.cpu_count()-1) * 2**6
n_images = None
save_data = False
min_clusters = 2
max_clusters = 7  # get 7 clusters, because we want 6 colours, and expect pure black to exist in the background.


home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')

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
        initialise = perf_counter()
        for i in range(0, len(image_paths), batch_size):
            start = perf_counter()
            batch_img_paths = image_paths[i:i+batch_size]
            batch_mask_paths = mask_paths[i:i+batch_size]
            # print(f'Batch {i//batch_size+1}: {len(batch_img_paths)} images and masks')
            batch_images = list(io_exec.map(combine_imgmask, batch_img_paths, batch_mask_paths))
            counter += len(batch_images)
            
            with concurrent.futures.ProcessPoolExecutor() as cpu_executor:
                batch_colours = list(cpu_executor.map(get_colours, batch_images))
            end = perf_counter() 
            print(f'{counter} / {len(mask_paths)}: {np.round(counter / len(mask_paths) * 100,2)}%, est time remaining: {np.round((end-start)/batch_size * (len(mask_paths) - counter),2)/60}m, total: {np.round(end-initialise,2)}s')
            # end = perf_counter()
            # print(f'Processed {counter} images in {end-start:.2f} seconds')
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
    # this will only show the last set of batch images
    # for i in range(0,len(batch_images)):
        # show(batch_images[i][0], batch_images[i][1])
   

if __name__ == '__main__':
    main()
