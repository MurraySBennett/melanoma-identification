import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import offsetbox
from sklearn.manifold import Isomap


from time import perf_counter
from glob import glob
from os import path
from cv_transforms import ABC_aligned, cv_btl_scale

show_fig=True

home=path.join(path.expanduser('~'), "win_home", "Documents", "melanoma-identification", "melnet")
paths = dict(
    figures=path.join(home, "figures"),
    data=path.join(home, "data")
)

files=sorted(glob(path.join(paths["data"], "*activations*csv")))
activations = pd.read_csv(files[0])
data = pd.read_csv(path.join(paths["data"], "btl-cv-data.csv"))
data = ABC_aligned(data)
data = cv_btl_scale(data, replace=True)
data = data[["id", "malignant", "pi_sym", "pi_bor", "pi_col", "sym", "bor", "col"]]
data = pd.merge(data, activations, on="id")


features_nn = ["n1", "n2", "n3", "n4"]
features_btl = ["pi_sym", "pi_bor", "pi_col"]
features_cv = ["sym", "bor", "col"]
malignant = ["malignant"]

print(data.head())

X = data[features_nn]
y = data[malignant].to_numpy().T[0]
label=data[["id"]].to_numpy().T[0]

n_neighbors = 10
n_components = 2

# start = perf_counter()

# iso_X = Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X).T
# print(iso_X.shape)
# iso_X = np.row_stack((iso_X, y)) # add malignancy for colouring
# print(iso_X.shape)

# end = perf_counter()
# print(end-start)

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# plt.scatter(iso_X[0], iso_X[1], c=iso_X[2], cmap='jet')
# plt.axis("tight")
# plt.show()


def plot_embedding(X):
    _, ax = plt.subplots()
    
    for img in images["id"]:
        ax.scatter(
        # data[label==data['id'][0]].values[0][:2]
            X[label==img[0]].values[0][:2],
            # *X[label==img, :2].T,
            marker=f"${img}$",
            s=10,
            alpha=0.4,
            zorder=2,
        )
    shown_images = np.array([[1., 1.]])
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images.img[i]), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)
    ax.axis("off")
                


