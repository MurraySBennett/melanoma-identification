# all melanoma
# no lesions with more than 1 picture (reduce to 1)
# use all the 2017 data -- they have 'ground truth' masks

## Note: See how fierce this filter is, see how your image segmentation analysis goes,
# see how that might filter, then see how they filter together

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

meta_df = pd.read_csv(os.path.join(os.getcwd(), "..", "images", "metadata.csv"))

# images with associated ground truth masks
gt_df = pd.read_csv(os.path.join(os.getcwd(), "..", "images", "ISIC-2017_Training_Part3_GroundTruth.csv"))

# melanoma images
mel_df = meta_df[meta_df['diagnosis'] == 'melanoma']
# unique melanoma lesions -- this ends up being very small (~600)
# umc = mel_df['lesion_id'].value_counts().reset_index()
# umc.columns = ["lesion_id", "count"]

image_id_list = pd.concat([gt_df['image_id'], mel_df['isic_id']])


n_images = meta_df.shape[0]

n_unique_lesions = meta_df.lesion_id.unique()
# unique_lesion_counts
ulc = meta_df['lesion_id'].value_counts().reset_index(name="count")
ulc.columns = ["lesion_id", "count"]

# unique patients
upc = meta_df['patient_id'].value_counts().reset_index()
upc.columns = ["patient_id", "count"]




# sns.catplot(data=ulc_hist, x='lesion_id', y='isic_id', kind='bar')
# plt.show()
