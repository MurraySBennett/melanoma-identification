import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
# import plotly.express as px
# from IPython.display import display, HTML

from PIL import Image
import cv2 as cv

from glob import glob
from os import path
from pathlib import PurePosixPath, PureWindowsPath
from cv_transforms import ABC_aligned, cv_btl_scale
from pprint import pprint as pp


home=path.join(path.expanduser('~'), "win_home", "Documents", "melanoma-identification", "melnet")
paths = dict(
    figures=path.join(home, "figures"),
    data=path.join(home, "data")
)

files=sorted(glob(path.join(paths["data"], "*embed*sansDCA*SNE*csv")))
n_dims = 2
activations = pd.read_csv(files[n_dims-2]) # 0 is 2 dims, 1 is 3

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

img_ids = data["id"].values
data.to_json(path.join(paths["figures"], "dcnn_dim_data.json"))


cor_mat = data[features_btl + features_cv].corr()
mask = np.triu(np.ones_like(cor_mat, dtype=bool))

show_fig=False
if show_fig:
    plt.figure(figsize=(12,10))
    sns.heatmap(cor_mat, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, mask=mask)
    plt.title("Correlation Heatmap")

    sns.pairplot(data[features_btl + features_cv + malignant], hue="malignant",corner=True)
    sns.pairplot(data[features_btl + features_nn + malignant], hue="malignant",corner=True)
    sns.pairplot(data[features_cv + features_nn + malignant], hue="malignant",corner=True)
    plt.show()


def append_images(df):
    img_ids = df["id"].values
    img_path = path.join(path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melxpertise", "melxpertise", "images")
    img_paths = [path.join(img_path, i + ".JPG") for i in img_ids]

    img_list = [cv.imread(i)[...,::-1] for i in img_paths]
    df["img"] = img_list
    return df


def get_images(df, var, pctile=95, n_images=5):
    from_top = True if pctile > 50 else False
    pctile_value = np.percentile(data[var], pctile)
    if from_top:
        filtered = df[df[var] >= pctile_value].reset_index(drop=True)
    else:
        filtered = df[df[var] <= pctile_value].reset_index(drop=True)
    imgs = filtered.sort_values(by=var, ascending=not from_top)
    imgs = imgs.sample(n=n_images).reset_index(drop=True).sort_values(by=var, ascending=not from_top).reset_index(drop=True)

    imgs = append_images(imgs)
    return imgs


def plot_stacks(img_dict):
    n_dims = len(img_dict['low'])
    n_columns = 4
    n_rows = 3
    fig, ax = plt.subplots(nrows=2, ncols=n_dims, figsize=(15,6))
    for i, (var, images) in enumerate(img_dict['high'].items()):
        row = i // n_columns
        col = i % n_columns

        stack = np.vstack(images['img'].values)
        ax[0, i].imshow(stack)
        ax[0, i].set_title(var)
        ax[0, i].axis('off')
    for i, (var, images) in enumerate(img_dict['low'].items()):
        stack = np.vstack(images['img'].values)
        ax[1, i].imshow(stack)
        ax[1, i].set_title(var)
        ax[1, i].axis('off')
    plt.tight_layout()


percentile = 95
n_images=5
img_ids = dict(low={}, high={})
for i, f in enumerate(features_nn):
    img_ids['low'][f]  = get_images(data, f, pctile=100-percentile, n_images=n_images) 
    img_ids['high'][f] = get_images(data, f, pctile=percentile, n_images=n_images)


if n_dims == 3:
    # fig = plt.figure(figsize=(20,20))
    # ax = fig.add_subplot(111, projection="3d")
    fig, ax = plt.subplots(figsize=(20,20))
else:
    fig, ax = plt.subplots(figsize=(20,20))

for i, level in enumerate(img_ids): # for low and high
    for j, var in enumerate(img_ids[level]): # for each df sorted by var j
        for k, row in img_ids[level][var].iterrows(): #plot each image
            img = row["img"]
            x, y = row["dim1"], row["dim2"]
            if n_dims == 3 and var=='dim3':
                imagebox = OffsetImage(img, zoom=0.2 if row["dim3"] <0 else 0.6)
                ab = AnnotationBbox(imagebox, (row['dim1'], row['dim2']), frameon=False, pad=0)
            else:
                imagebox = OffsetImage(img, zoom=0.25)
                ab = AnnotationBbox(imagebox, (row['dim1'], row['dim2']), frameon=False, pad=0)
            ax.add_artist(ab)

ax.set_xlim(data['dim1'].min() - 1, data['dim1'].max() + 1)
ax.set_ylim(data['dim2'].min() - 1, data['dim2'].max() + 1)
if n_dims ==3:
    ax.set_ylim(data['dim3'].min() - 1, data['dim3'].max() + 1)
ax.axis("off")

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
plot_stacks(img_ids)

plt.show()

