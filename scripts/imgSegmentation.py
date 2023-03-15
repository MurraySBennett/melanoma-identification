import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from ColourBalance import *
import multiprocessing
import time

n = 1  # number of images to process
size = (512, 384)

def display(img, labels, shape_factor):
    plt.figure(figsize=(len(img*2), 3))
    for i in range(len(img)):
        plt.subplot(1, len(img), i+1)
        if np.max(img[i]) == 1:
            plt.imshow(img[i], cmap='gray')
        else:
            plt.imshow(img[i])
        plt.axis('off')
        plt.title(labels[i])
        if i == len(img):
            plt.text(1, 1, f'Shape Factor: {shape_factor}')
    plt.tight_layout()
    plt.show()


def read_data(n_img):
    df = pd.read_csv(os.path.join(os.getcwd(), "images", "ISIC-database", "metadata.csv"), sep=",")
    df.rename(columns={"isic_id": "image_id"}, inplace=True)
    mal_df, ben_df = [x for _, x in df.groupby(df["benign_malignant"].str.contains(r"benign", na=True))]

    mal_df = mal_df.iloc[:n_img].reset_index()
    ben_df = ben_df.iloc[:n_img].reset_index()
    return df, mal_df, ben_df


def img_seg(img, size):
    cb = colour_balance()  # bad to instantiate this each time the function is called.
    # crop = crop_img(img, 0.05)  # proportion of image cropped
    rsz = cv2.resize(img, size)
    # r_norm = red_channel(rsz)

    col_adj = cb.automatic_brightness_and_contrast(rsz, 1)
    col_bal = cb.simplest_cb(col_adj, 1)

    gray, blkhat, thresh2, processed = process_img(rsz, size)  # not including the r_nrom, col_adj or col_bal process at the moment.
    # crop = [crop_img(i, size) for i in dst]

    mask, contour, contoured_img, SV_conversion = hsv_channel(processed)
    masked_img = impose_mask(rsz, mask)

    # I don't think I use this bit anymore
    # otsu, cv = edge_detection(dst)
    # sp = [super_pixel(i, n_segments=5) for i in dst]

    return rsz, col_adj, col_bal, processed, masked_img, mask, contour, contoured_img, SV_conversion


data, mal, ben = read_data(n)

img_path = os.path.join(os.getcwd(), 'images', 'ISIC-database')
extension = '.JPG'

mal['og'] = [cv2.imread(os.path.join(img_path, i + extension)) for i in mal['image_id']]
mal['rsz'], mal['col_adj'], mal['col_bal'], mal['proc'], mal['masked'], mal['mask'], mal['contour'], mal['contoured'], mal['SV'] = map(list, zip(*[img_seg(i, size) for i in mal['og']]))
mal['shape_factor'] = [get_shape_factor(c) for c in mal['contour']]



ben['og'] = [cv2.imread(os.path.join(img_path, i + extension)) for i in ben['image_id']]
ben['rsz'], ben['col_adj'], ben['col_bal'], ben['proc'], ben['masked'], ben['mask'], ben['contour'], ben['contoured'], ben['SV'] = map(list, zip(*[img_seg(i, size) for i in ben['og']]))




# mal['rsz'] = [cv2.resize(i, size) for i in mal['og']]
# mal['col_adj'] = [cb.automatic_brightness_and_contrast(i, 1) for i in mal['rsz']]
# mal['col_bal'] = [cb.simplest_cb(i, 1) for i in mal['col_adj']]
#
# mal['gray'], mal['blkhat'], mal['thresh2'], mal['dst'] = map(list, zip(*[process_img(i, size) for i in mal['col_bal']]))
# # mal['crop'] = [crop_img(i, size) for i in mal['dst']]

# mal['otsu'], mal['cv'] = map(list, zip(*[edge_detection(i) for i in mal['dst']])) #dst
# # mal['sp'] = [super_pixel(i, n_segments=5) for i in mal['dst']]
# mal['masked'] = [impose_mask(mal['rsz'][i], mal['cv'][i]) for i in range(len(mal['og']))]


labels = ["Original", "Processed", "HSV Conversion", "Contour Extraction", "Segmented Lesion"]
for i in range(n):
    img = [mal['og'][i], mal['proc'][i], mal['SV'][i], mal['contoured'][i], mal['masked'][i]]
    shape_f = mal['shape_factor'][i]
    display(img, labels, shape_f)
