from os import path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pprint import pprint as pp
from PIL import Image

from glob import glob
from pathlib import PurePosixPath, PureWindowsPath
from cv_transforms import ABC_aligned, cv_btl_scale

home=path.join(path.expanduser('~'), "win_home", "Documents", "melanoma-identification", "melnet")
paths = dict(
    figures=path.join(home, "figures"),
    data=path.join(home, "data"),
    images=path.join(path.expanduser("~"), "win_home", "melanoma-identification", "images", "resized")
    )

font = FontProperties()
font_colour = "black"
font_size = 20
line_colour = "black"
axis_label_font_size = 18

levels = ['High', 'Low']
dim_labels = ["Dimension 1", "Dimension 2", "Dimension 3"]

n_dims = 3
files=sorted(glob(path.join(paths["data"], "*embed*" + str(n_dims) + "*SNE*csv")))

def get_ids(data, feature, n_images):
    data = data.sort_values(by=feature, ascending=True)
    data = data[['id', feature]]
    data = data.dropna().reset_index(drop=True)

    high = data['id'].head(n_images)
    low  = data['id'].tail(n_images)
    data = list(pd.concat([high, low]))

    return [data]

def load_img(f, img_path):
    img_path = path.join(img_path, f)
    img = cv2.imread(img_path+".JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def no_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def clear_ticks(ax):
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)

for f in files:
    model_name = "default"
    if "sansDCA" in f:
        model_name += "_sansDCA"
    if "sansPatch" in f:
        model_name += "_sansPatch"

    if "sansPatch" not in model_name:
        print(f"skipping {f}")
        continue

    print(f"loading {f}")
    activations = pd.read_csv(f)
    activations = activations.drop(columns=["malignant"])
    data = pd.read_csv(path.join(paths["data"], "btl-cv-data.csv"))
    data = ABC_aligned(data)
    data = cv_btl_scale(data, replace=True)
    data = data[["id", "malignant", "pi_sym", "pi_bor", "pi_col", "sym", "bor", "col"]]
    data = pd.merge(data, activations, on="id")

    features_nn = ["dim1", "dim2", "dim3"]  
    features_nn = features_nn[:n_dims]

    features_btl = ["pi_sym", "pi_bor", "pi_col"]
    features_cv = ["sym", "bor", "col"]
    malignant = ["malignant"]

    n_per_col = 30#125 
    n_sub_rows= 80#75

    # you're now taking the top and tail and plotting them all together, so halve the value.
    n_images = (n_per_col * n_sub_rows) // 2

    exemplars = {}
    for feature in features_nn:
        try:
            egs = get_ids(data, feature, n_images=n_per_col * n_sub_rows)
            exemplars[feature] = egs
        except:
            continue

    n_rows = 1
    n_cols = 1

    for i, feature in enumerate(features_nn):
        fig, axs = plt.subplots(n_rows, n_cols,figsize=(16,8))
        ids = exemplars[feature][0]
        stacked_images = []
        for img_id in ids:
            img = load_img(img_id, paths['images'])
            stacked_images.append(img)

        num_rows = len(ids) // n_per_col
        remainder = len(ids) % n_per_col
        rows = []
        for k in range(num_rows):
            row_images = stacked_images[k*n_per_col: (k+1)*n_per_col]
            # row = np.hstack(row_images)
            row = np.vstack(row_images)
            rows.append(row)
        if remainder > 0:
            last_row_images = stacked_images[-remainder:]
            # last_row = np.hstack(last_row_images)
            last_row = np.vstack(last_row_images)
            rows.append(last_row)

        # final_image = np.vstack(rows)
        final_image = np.hstack(rows)
        axs.imshow(final_image)

        axs.tick_params(labelbottom=False, labelleft=False)
        no_spines(axs)
        clear_ticks(axs)

        axs.set_title(feature, font=font, fontsize=font_size, color=font_colour)

    # for j, level in enumerate(levels):
    #     axs[j,0].set_ylabel(level, font=font, fontsize=axis_label_font_size, color=font_colour)

        plt.tight_layout()
        # plt.suptitle(model_name)
        # if model_name is not None:
            # dim1_high_patches = pd.DataFrame(exemplars["dim1"][0])
            # dim1_high_patches.columns = ["id"]
            # dim1_high_patches.to_csv(path.join(paths["data"], 'patch_image_ids.txt'), sep=',', index=False)
            # plt.savefig('vis_melnet_dims_sansPatch.pdf', format='pdf', dpi=600, bbox_inches='tight')

        # else:
            # plt.savefig('vis_melnet_dims.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path.join(paths["figures"], f'vis_melnet_{model_name}_{feature}_.pdf'), format='pdf', dpi=600, bbox_inches='tight')

        plt.show()


