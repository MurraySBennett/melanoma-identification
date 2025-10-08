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
    images=path.join(path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melxpertise", "melxpertise", "images")
    )

font = FontProperties()
font_colour = "black"
font_size = 20
line_colour = "black"
axis_label_font_size = 18

levels = ['High', 'Middle', 'Low']
dim_labels = ["Dimension 1", "Dimension 2", "Dimension 3"]

n_dims = 3
# files=sorted(glob(path.join(paths["data"], "*embed*sansDCA*" + str(n_dims) + "*SNE*csv")))
files=sorted(glob(path.join(paths["data"], "*embed*" + str(n_dims) + "*SNE*csv")))
for f in files:
    model_name = "sansDCA" if "sansDCA" in f else None
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


    def get_ids(data, feature, n_images):
        data = data.sort_values(by=feature)
        data = data[['id', feature]]
        data = data.dropna().reset_index(drop=True)

        high = list(data.tail(n_images)['id'])
        low = list(data.head(n_images)['id'])
        # high = list(data.tail(n_images*3)['id'].head(n_images))
        # low = list(data.head(n_images*3)['id'].head(n_images))

        mean = data[feature].mean()
        medium = list(data.iloc[(data[feature] - mean).abs().argsort()[:n_images]]['id'])
        return [high, medium, low]

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

    n_per_col = 5
    n_sub_rows= 4

    exemplars = {}
    for feature in features_nn:
        try:
            egs = get_ids(data, feature, n_images=n_per_col * n_sub_rows)
            exemplars[feature] = egs
        except:
            continue
    pp(exemplars)

    n_rows = 3
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols,figsize=(16,8))

    for i, feature in enumerate(features_nn):
        for j, level in enumerate(levels):
            ids = exemplars[feature][j]
            print(f"ids: {ids}, level: {level}, feature: {feature}")
            stacked_images = []
            for img_id in ids:
                img = load_img(img_id, paths['images'])
                stacked_images.append(img)

            num_rows = len(ids) // n_per_col
            remainder = len(ids) % n_per_col
            rows = []
            for k in range(num_rows):
                row_images = stacked_images[k*n_per_col: (k+1)*n_per_col]
                row = np.hstack(row_images)
                rows.append(row)
            if remainder > 0:
                last_row_images = stacked_images[-remainder:]
                last_row = np.hstack(last_row_images)
                rows.append(last_row)

            final_image = np.vstack(rows)
            axs[j, i].imshow(final_image)

            axs[j, i].tick_params(labelbottom=False, labelleft=False)
            no_spines(axs[j, i])
            clear_ticks(axs[j,i])

    for i , feature in enumerate(dim_labels):
        axs[0, i].set_title(feature, font=font, fontsize=font_size, color=font_colour)

    for j, level in enumerate(levels):
        axs[j,0].set_ylabel(level, font=font, fontsize=axis_label_font_size, color=font_colour)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    if model_name is not None:
        plt.suptitle(model_name)
        plt.savefig('vis_melnet_dims_sansDCA.pdf', format='pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig('vis_melnet_dims.pdf', format='pdf', dpi=600, bbox_inches='tight')

plt.show()

