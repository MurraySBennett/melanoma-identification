import sys
from os import path
sys.path.append(path.join(path.expanduser('~'), 'win_home', 'melanoma-identification', 'machine-learning'))
from paths import paths
import pandas as pd

save_data = True

exp_imgs = pd.read_csv('feature-rating-image-list.txt', delim_whitespace=True, header=0)
exp_imgs['id'] = [x.strip('.JPG') for x in exp_imgs['id']]

malignant_ids = pd.read_csv('malignant_ids.txt', header=0)

shape = pd.read_csv('shape.txt', delim_whitespace=True, header=0)
shape['id'] = [x.strip('.png') for x in shape['id']]

colour = pd.read_csv('colours_continuous.txt', delim_whitespace=True, header=0)
colour['id'] = [x.strip('.JPG') for x in colour['id']]

data = exp_imgs\
        .merge(malignant_ids, on='id', how='inner')\
        .merge(shape, on='id', how='inner')\
        .merge(colour, on='id', how='inner')

if save_data:
    data.to_csv('cv-data.csv', index=False)
