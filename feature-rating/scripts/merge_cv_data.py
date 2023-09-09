import sys
from os import path
import pandas as pd

save_data = True

home = "/mnt/c/Users/qlm573/melanoma-identification/"
data_path = path.join(home, "feature-rating", "cv-data")
exp_imgs = pd.read_csv(path.join(data_path, 'feature-rating-image-list.txt'), delim_whitespace=True, header=0)
exp_imgs['id'] = [x.strip('.JPG') for x in exp_imgs['id']]

malignant_ids = pd.read_csv(path.join(data_path, 'malignant_ids.txt'), header=0)

shape = pd.read_csv(path.join(data_path, 'shape.txt'), delim_whitespace=True, header=0)
shape['id'] = [x.strip('.png') for x in shape['id']]

colour = pd.read_csv(path.join(data_path, 'colours_continuous.txt'), delim_whitespace=True, header=0)
colour['id'] = [x.strip('.JPG') for x in colour['id']]

data = exp_imgs\
        .merge(malignant_ids, on='id', how='inner')\
        .merge(shape, on='id', how='inner')\
        .merge(colour, on='id', how='inner')

if save_data:
    data.to_csv(path.join(data_path, 'cv-data.csv'), index=False)

