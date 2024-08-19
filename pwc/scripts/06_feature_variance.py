from pathlib import Path
from matplotlib.font_manager import FontProperties
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

home        = Path(__file__).resolve().parent.parent
data_path   = home / "data" / "estimates" / "btl-cv-data.csv"
img_path    = home.parent / "images" / "resized"
figure_path = home / "figures"
data = pd.read_csv(data_path)

features = ['pi_sym', 'pi_bor', 'pi_col']
data = data[['id', 'malignant'] + features].dropna()

feature_labels = ["Asymmetry", "Border\nIrregularity", "Colour\nVariance"]
exemplars = {}

# font = FontProperties(fname="Garamond BoldCondensed.ttf")
font = FontProperties()
FONT_COLOUR = "black" #"#0c2340" # the hex code for contour_colour
FONT_SIZE = 24
LINE_COLOUR = "black"#"#D4440D"
AXIS_LABEL_FONT_SIZE = 18

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42


# only malignant?
# data = data[data["malignant"] == 1].reset_index()
def main():
    n_images = 5
    pctiles = np.linspace(0.1, 99.9, n_images)#[20, 45, 60, 75, 80]
    print(f"Images drawn from around the {pctiles} pctiles")

    for feature in features:
        try:
            egs = [get_pctile(feature, p, 0 if p < 67 else 1) for p in pctiles]
            exemplars[feature] = egs
        except:
            continue

    n_rows = 3
    n_cols = 1
    _, axs = plt.subplots(n_rows, n_cols, figsize=(16,8))

    for i, feature in enumerate(features):
        final_image = np.hstack(exemplars[feature])
        axs[i].imshow(final_image)
        axs[i].tick_params(labelbottom=False, labelleft=False)
        no_spines(axs[i])
        clear_ticks(axs[i])

    axs[0].set_title(
        r'Less $\longleftarrow \qquad\qquad\qquad\qquad\qquad\qquad\qquad \longrightarrow$ More',
        font=font, fontsize=FONT_SIZE, color=FONT_COLOUR
    )

    for i , feature in enumerate(feature_labels):
        axs[i].set_ylabel(
            feature,
            font=font, fontsize=FONT_SIZE, color=FONT_COLOUR
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(
        figure_path / "feature_variance.pdf",
        format="pdf", dpi=600, bbox_inches="tight"
    )
    plt.savefig(
        figure_path / "feature_variance.png",
        format="png", dpi=600, bbox_inches="tight"
    )
    plt.show()


def add_border(img, size, colour=(255, 255, 255)):
    """ add {colour} border to images for display"""
    img[0:size, :] = colour
    img[-size:, :]  = colour
    img[:, 0:size] = colour
    img[:, -size:]  = colour
    return img


def load_img(f, im_path):
    """ load image and add a border """
    im_path = im_path / f"{f}.JPG"
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_border(img, 3)
    return img


def get_pctile(f, p, malignant = 0):
    """ return image from estimate value percentile """
    not_f = features.copy()
    size = 0
    buffer = 0.05
    while size == 0:
        img_id = data[
                (data[f] > np.percentile(data[f], p)) &
                (data[f] < np.percentile(data[f], p+buffer)) &
                (data["malignant"] == malignant)
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

# def get_ids(data, feature, n_images):
#     data = data.sort_values(by=feature)
#     data = data[['id', feature]]
#     data = data.dropna().reset_index(drop=True)
    # avoid the extreme-extremes because they contain images that might better have been excluded
#     high = list(data.tail(n_images*3)['id'].head(n_images))
#     low = list(data.head(n_images*3)['id'].head(n_images))
#     mean = data[feature].mean()
#     medium = list(data.iloc[(data[feature] - mean).abs().argsort()[:n_images]]['id'])
#     return [high, medium, low]

def no_spines(ax):
    """ Remove axis spines """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def clear_ticks(ax):
    """ remove axis ticks """
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)


if __name__ == "__main__":
    main()
