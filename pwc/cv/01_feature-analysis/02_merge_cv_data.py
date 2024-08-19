import sys
from os import path
import pandas as pd

save_data = True
here = path.dirname(path.abspath(__file__))

exp_imgs        = pd.read_csv(path.join(here, 'sampled_shape_ids.txt'), delim_whitespace=True, header=0)
malignant_ids   = pd.read_csv(path.join(here, 'malignant_ids.txt'), header=0)
shape           = pd.read_csv(path.join(here, 'cv_shape.txt'), delim_whitespace=True, header=0)
shape['id']     = [x.strip('.png') for x in shape['id']]
colour          = pd.read_csv(path.join(here, 'cv_colours_continuous.txt'), delim_whitespace=True, header=0)
colour['id']    = [x.strip('.JPG') for x in colour['id']]

data = exp_imgs\
        .merge(malignant_ids, on='id', how='inner')\
        .merge(shape, on='id', how='inner')\
        .merge(colour, on='id', how='inner')

if save_data:
    data.to_csv(path.join(here, "..", "..", data, "estimates", "cv-data.csv"), index=False)

