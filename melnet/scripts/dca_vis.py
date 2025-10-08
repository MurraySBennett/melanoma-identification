from os import path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

data_path = path.join(path.expanduser("~"), "win_home", "Documents", "melanoma-identification", "melnet", "data", "dca_images.txt")
img_path = path.join(path.expanduser("~"), "win_home", "melanoma-identification", "images", "resized")
data = pd.read_csv(data_path)

levels = ['High', 'Middle', 'Low']
exemplars = {}

# font = FontProperties(fname="Garamond BoldCondensed.ttf")
font = FontProperties()
font_colour = "black" #"#0c2340" # the hex code for contour_colour
font_size = 20
line_colour = "black"#"#D4440D"
axis_label_font_size = 18

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
# 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42


def get_ids(data, n_images):
    data = data.sort_values(by="prop")
    # data = data[['id', feature]]
    data = data.dropna().reset_index(drop=True)
    high = list(data.tail(n_images*3)['id'].head(n_images)) # avoiding the extreme-extremes because they contain images that might better have been excluded
    low = list(data.head(n_images*3)['id'].head(n_images))
    mean = data["prop"].mean()
    medium = list(data.iloc[(data["prop"] - mean).abs().argsort()[:n_images]]['id'])
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


# for feature in features:
n_per_row = 8
n_sub_rows = 3
try:
    egs = get_ids(data, n_images=n_per_row * n_sub_rows)
    exemplars["dca_prop"] = egs
except:
    print("error getting examples")

n_rows = 3
n_cols = 1
fig, axs = plt.subplots(n_rows, n_cols,figsize=(16,8))

for j, level in enumerate(levels):
    ids = exemplars["dca_prop"][j]
    print(f"ids: {ids}, level: {level}")
    stacked_images = []
    for img_id in ids:
        img = load_img(img_id, img_path)
        stacked_images.append(img)
    num_rows = len(ids) // n_per_row
    remainder = len(ids) % n_per_row
    rows = []
    for i in range(num_rows):
        row_images = stacked_images[i*n_per_row: (i+1)*n_per_row]
        row = np.hstack(row_images)
        rows.append(row)

    if remainder > 0:
        last_row_images = stacked_images[-remainder:]
        last_row = np.hstack(last_row_images)
        rows.append(last_row)

    final_image = np.vstack(rows)

    # img_1 = load_img(ids[0], img_path)
    # img_2 = load_img(ids[1], img_path)
    # img_3 = load_img(ids[2], img_path)
    # img_4 = load_img(ids[3], img_path)
    # img_5 = load_img(ids[4], img_path)
    # img_6 = load_img(ids[5], img_path)

    # img1 = np.hstack((img_1, img_2, img_3))
    # img2 = np.hstack((img_4, img_5, img_6))

    # img = np.vstack((img1, img2))

    axs[j].imshow(final_image) #img)
    axs[j].tick_params(labelbottom=False, labelleft=False)
    no_spines(axs[j])
    clear_ticks(axs[j])
    # axs[j, i].axis('off')

# for i , feature in enumerate(feature_labels):
#     axs[0, i].set_title(feature, font=font, fontsize=font_size, color=font_colour)

for j, level in enumerate(levels):
    axs[j].set_ylabel(level, font=font, fontsize=axis_label_font_size, color=font_colour)

# line1 = plt.Line2D((0.349, 0.349), (0.02, 0.98), color=line_colour, linewidth=2.2)
# line2 = plt.Line2D((0.671, 0.671), (0.02, 0.98), color=line_colour, linewidth=2.2)
# line3 = plt.Line2D((0.025, 0.99), (0.34, 0.34), color=line_colour, linewidth=2, linestyle=":")
# line4 = plt.Line2D((0.025, 0.99), (0.64, 0.64), color=line_colour, linewidth=2, linestyle=":")
# lines = [line3, line4]#, line3, line4]
# [fig.add_artist(l) for l in lines]

plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig('dca_vis.pdf', format='pdf', dpi=600, bbox_inches='tight')


plt.figure()
n_bins = 45
plt.hist(data["prop"], bins=n_bins, color='blue', alpha=0.7)
plt.xlabel('Proportion of Black Pixels')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('dca_prop_dist.pdf', format='pdf', dpi=600, bbox_inches='tight')

plt.show()
