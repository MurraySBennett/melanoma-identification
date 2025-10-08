import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2 as cv

from glob import glob
from os import path
from cv_transforms import ABC_aligned, cv_btl_scale
from pprint import pprint as pp


home=path.join(path.expanduser('~'), "win_home", "Documents", "melanoma-identification", "melnet")
paths = dict(
    figures=path.join(home, "figures"),
    data=path.join(home, "data")
)



files=sorted(glob(path.join(paths["data"], "*embed*SNE*csv")))
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

nth_pts = 10
data = data[::nth_pts]
print(data.shape)
cor_mat = data[features_btl + features_cv].corr()
mask = np.triu(np.ones_like(cor_mat, dtype=bool))

show_fig=True

if show_fig:
    plt.figure(figsize=(12,10))
    sns.heatmap(cor_mat, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, mask=mask)
    plt.title("Correlation Heatmap")
    plt.savefig(path.join(paths["figures"], 'btl-cv-cor-heatmap.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    sns_plt = sns.pairplot(data[features_btl + features_cv + malignant], hue="malignant",corner=True)
    sns_plt.savefig(path.join(paths["figures"], 'btl-cv-cor-full.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    sns_plt = sns.pairplot(data[features_btl + features_nn + malignant], hue="malignant",corner=True)
    sns_plt.savefig(path.join(paths["figures"], 'btl-nn-cor-full.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    sns_plt = sns.pairplot(data[features_cv + features_nn + malignant], hue="malignant",corner=True)
    sns_plt.savefig(path.join(paths["figures"], 'cv-nn-cor-full.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    plt.show()


