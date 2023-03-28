import os 
from os import path
import pandas as pd
import glob
from pprint import pprint

from minimise_dca import * 
from image_processes import process_img, colour_cluster, my_clahe, otsu_segment, grabcut_segment, display
from file_management import save_img

K = 6 # n clusters
n_images = 1
##### --> set paths and filenames
home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
paths = dict(
    home=home_path,
    images=path.join(home_path, "images", "resized"),
    masks=path.join(home_path, "images", "segmented", "masks"),
    segmented=path.join(home_path, "images", "segmented", "images"),
    data=path.join(home_path, "images", "metadata")
    )
image_paths = glob.glob(paths["images"] + "/*.JPG")
image_ids = [i.split("/")[-1] for i in image_paths]

##### --> define functions

def main():
    data = pd.read_csv(paths["data"] + "/metadata.csv")
    data["id_int"] = [i.split("_")[1] for i in data["isic_id"]]
    data = data.sort_values(by=["id_int"], ascending=True).reset_index()
    if n_images is not None:
        data = data.head(n_images)

    data["image_raw"] = [cv.imread(path.join(paths["images"], i + ".JPG")) for i in data["isic_id"]]
    data["dca"] = [get_dca(i) for i in data["image_raw"]]

    data["dca_rm"] = [paint_dca(image=data["dca"][i][0], mask=data["dca"][i][1], method='inpaint_ns') for i in range(len(data["image_raw"]))]
    data["image_proc"] = [process_img(i) for i in data["dca_rm"]]
    # data["image_masks"], data["contours"], data["contoured_image"] = map(list,
    #         zip(*[segment(i) for i in
    #     data["image_proc"]]))
    # data["image_masked"] = [cv.bitwise_and(data["image_proc"][i], data["image_proc"][i],
    #     mask=data["image_masks"][i].astype(np.uint8)) for i in
    #     range(len(data["image_proc"]))]
    # shape_factor = [get_shape_factor(c) for c in data["contours"]]
    data["kmeans_img"] = [colour_cluster(i, n_clusters=K) for i in data["image_proc"]]
    ###### adaptive histogram equalisation -- contrast limited (clahe)
    data["enhanced"] = [my_clahe(i) for i in data["kmeans_img"]]
    data["otsu"] = [otsu_segment(data["image_raw"][i], data["enhanced"][i]) for i in range(len(data["enhanced"]))]
    data["grabcut"] = [grabcut_segment(data["image_raw"][i], data["enhanced"][i]) for i in range(len(data["enhanced"]))]

    # [display( [data["image_raw"][i], data["image_proc"][i],  data["otsu"][i]], ["raw", "hair removal", "segmented"]) for i in range(len(data["image_raw"]))]
    
    [save_img(data["otsu"][i], paths["masks"], data["isic_id"][i]) for i in range(len(data["isic_id"]))]

if '__name__':
    main()
