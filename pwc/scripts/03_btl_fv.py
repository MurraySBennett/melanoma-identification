from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# font = FontProperties(fname="Garamond BoldCondensed.ttf")
font = FontProperties()
FONT_COLOUR = "black" #/"#0c2340" # the hex code for contour_colour
FONT_SIZE   = 20
LINE_COLOUR = "black"#"#D4440D"
AXIS_LABEL_FONT_SIZE = 18
plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42

home        = Path(__file__).resolve().parent.parent 
data_path   = home / "data" / "estimates" / "btl_cv_data.csv"

img_path    = home.parent / "images" / "resized"
fig_path    = home / "figures"
data = pd.read_csv(data_path)

features    = ['pi_sym', 'pi_bor', 'pi_col']
data        = data[['id'] + features]
levels      = ['High', 'Middle', 'Low']
feature_labels = ["Asymmetry", "Border Irregularity", "Colour Variance"]
exemplars   = {}


def get_ids(d, f, n_images):
    d = d.sort_values(by=f)
    d = d[['id', f]]
    d = d.dropna().reset_index(drop=True)
    # avoiding the extremes because they contain images that might better have been excluded
    high    = list(d.tail(n_images*3)['id'].head(n_images))
    low     = list(d.head(n_images*3)['id'].head(n_images))
    mean    = d[f].mean()
    medium  = list(
        d.iloc[(d[f] - mean).abs().argsort()[:n_images]]['id']
    )
    return [high, medium, low]


def load_img(f, im_path):
    im_path = im_path / f"{f}.JPG"
    im         = cv2.imread(im_path)
    im         = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def no_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def clear_ticks(ax):
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)


N_PER_COL = 4
N_SUB_ROWS= 3
N_ROWS = 3
N_COLS = 3
for feature in features:
    egs = get_ids(data, feature, n_images=N_PER_COL * N_SUB_ROWS)
    exemplars[feature] = egs
fig, axs = plt.subplots(N_ROWS, N_COLS,figsize=(16,8))
for i, feature in enumerate(features):
    for j, level in enumerate(levels):
        ids = exemplars[feature][j]
        print(f"ids: {ids}, level: {level}, feature: {feature}")
        stacked_images = []
        for img_id in ids:
            img = load_img(img_id, img_path)
            stacked_images.append(img)

        num_rows = len(ids) // N_PER_COL
        remainder = len(ids) % N_PER_COL
        rows = []
        for k in range(num_rows):
            row_images = stacked_images[k*N_PER_COL: (k+1)*N_PER_COL]
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

for i , feature in enumerate(feature_labels):
    axs[0, i].set_title(feature, font=font, fontsize=FONT_SIZE, color=FONT_COLOUR)

for j, level in enumerate(levels):
    axs[j,0].set_ylabel(level, font=font, fontsize=AXIS_LABEL_FONT_SIZE, color=FONT_COLOUR)


plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0.02)



plt.savefig(
    fig_path / "btl_facevalidity.pdf",
    format='pdf', dpi=600, bbox_inches='tight'
)

