from os import path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pprint import pprint as pp

data_path   = path.join(path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melnet", "data", "btl-cv-data.csv")
img_path    = path.join(path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melxpertise", "melxpertise", "images")
data        = pd.read_csv(data_path)
nn_data     = pd.read_csv( path.join(path.expanduser("~"),
            "win_home", "Documents", "melanoma-identification", "melxpertise", "melxpertise", "feature_data", "svd_data.csv")
        )

features = ['pi_sym', 'pi_bor', 'pi_col']
data = data[['id', 'malignant'] + features].dropna()
nn_features = ["svd1", "svd2"]

# only malignant?
# data = data[data["malignant"] == 1].reset_index()
# pp(data.head())

def add_border(img, size, colour=(255, 255, 255)):

    img[0:size, :] = colour
    img[-size:, :]  = colour
    img[:, 0:size] = colour
    img[:, -size:]  = colour
    return img

def load_img(f, img_path):
    img_path = path.join(img_path, f) + ".JPG"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_border(img, 3)
    return img

def get_pctile(d, f, p, malignant = 0):
    size = 0
    buffer = 0.05
    while size == 0:
        img_id = d[
                (d[f] > np.percentile(d[f], p)) & 
                (d[f] < np.percentile(d[f], p+buffer)) &
                (d["malignant"] == malignant)
                ]
        size = img_id.size
        buffer += 0.01

    img_id = list(img_id.head(1)['id'])
    # value = list(img_id.head(1).[f])
    img = load_img(img_id[0], img_path)
    return img
    # result = dict(
    #         img = img,
    #         img_id=image_id, 
    #         pctile=p, 
    #         buffer=buffer, 
    #         value=value
    #     )
    # return result

feature_labels  = ["Asymmetry", "Border\nIrregularity", "Colour\nVariance"]
nn_labels       = ["Feature 1", "Feature 2"]
exemplars, nn_exemplars = {}, {}

# font = FontProperties(fname="Garamond BoldCondensed.ttf")
font = FontProperties()
font_colour = "black" #"#0c2340" # the hex code for contour_colour
font_size = 24
line_colour = "black"#"#D4440D"
axis_label_font_size = 18

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
# 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42


def no_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def clear_ticks(ax):
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)


n_images = 3 
pctiles = np.linspace(2.5, 97.5, n_images*2)
# pctiles = np.concatenate((
#         np.linspace(5.5, 10, n_images),
#         np.linspace(90, 95.5, n_images)
#         ))
# pp(pctiles)
for feature in features:
    try:
        egs = [get_pctile(data, feature, p, 0 if p < 67 else 1) for p in pctiles]
        exemplars[feature] = egs
    except:
        continue

for feature in nn_features:
    try:
        egs = [get_pctile(nn_data, feature, p) for p in pctiles]
        nn_exemplars[feature] = egs
    except:
        continue


def plt_exemplars(images, feature_ids, labels, save_name, n_images):
    n_rows = len(feature_ids)
    n_cols = 1
    img_size=3
    width = img_size*n_images
    height= img_size*n_rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))

    for i, feature in enumerate(feature_ids):
        final_image = np.hstack(images[feature])
        axs[i].imshow(final_image)
        axs[i].tick_params(labelbottom=False, labelleft=False)
        no_spines(axs[i])
        clear_ticks(axs[i])

    axs[0].set_title(r'Less $\longleftarrow \qquad\qquad\qquad\qquad\qquad\qquad\qquad \longrightarrow$ More', font=font, fontsize=font_size, color=font_colour)
    for i , feature in enumerate(labels):
        axs[i].set_ylabel(feature, font=font, fontsize=font_size, color=font_colour)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig('../figures/'+save_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig('../figures/'+save_name+'.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()

# plt_exemplars(exemplars, 'feature_variance')
plt_exemplars(nn_exemplars, nn_features, nn_labels, 'svd_feature_variance', n_images*2)
