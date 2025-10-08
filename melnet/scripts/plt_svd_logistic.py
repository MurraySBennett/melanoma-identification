import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import pandas as pd
from pprint import pprint as pp
import cv2

from os import path
from glob import glob
from matplotlib.font_manager import FontProperties

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

def no_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def clear_ticks(ax):
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)


save_data = True
if save_data:
    data = pd.read_csv(path.join(paths["data"], "best_sansDCA_sansPatch_EfficientNetB0-GAPactivations.csv"))

    n_dims = 3
    # U, S, V_t = svd(data.drop(columns=["id", "malignant"]), full_matrices=False)
    # print(f'data: {data.shape}, U: {U.shape}, Vt: {V_t.shape}')
    # S = np.diag(S)
    # data["svd"] = U[:,:n_dims] @ S[0:n_dims, :n_dims] @ V_t[:n_dims,:] # the first column represents the most important dimension in terms of variance -- you could selec
    # data["svd"] = U[:,0] # the first column represents the most important dimension in terms of variance -- you could selec


    t_svd = TruncatedSVD(n_components=n_dims, n_iter=7, random_state=42)
    svd = t_svd.fit_transform(data.drop(columns=["id", "malignant"]))
    
    data["svd1"] = svd[:, 0]
    data["svd2"] = svd[:, 1]
    data["svd3"] = svd[:, 2]

    data = data[["id", "svd1", "svd2", "svd3", "malignant"]]
    data.to_csv(path.join(paths["data"], 'svd_data.csv'), index=False)
else:
    data = pd.read_csv(path.join(paths["data"], 'svd_data.csv'))

X = data['svd1'].values.reshape(-1,1)
y = data['malignant']

clf = LogisticRegression()
clf.fit(X,y)

X_test = np.linspace(X.min(), X.max(), 300)[:, np.newaxis]
loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()

# fig, axs = plt.subplots(n_rows, n_cols,figsize=(8,6))
plt.figure(1, figsize=(4,3))
plt.clf()
# axs.tick_params(labelbottom=False, labelleft=False)
# plt.scatter(data['svd'], data['malignant'])
# plt.plot(X_test, y_proba[:,1], color='black')
# plt.log_reg()
plt.scatter(data['svd1'], data['malignant'])
plt.plot(X_test, loss, label="LogReg Model", color="black")


plt.xlabel("SVD Values")
plt.ylabel("Malignancy")
# plt.yticks(range(0,1,0.25))
plt.ylim(0,1)
plt.legend(loc="upper left", fontsize="small")
plt.title("SVD Categorisation", font=font, fontsize=font_size, color=font_colour)
plt.tight_layout()
# plt.savefig(path.join(paths["figures"], 'vis_melnet_svd_dim.pdf'), format='pdf', dpi=600, bbox_inches='tight')

plt.show()






