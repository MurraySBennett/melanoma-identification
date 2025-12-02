import sys
from os import path
import pandas as pd
import glob
from ...config import (PATHS, FILES)

save_data = True
here = path.dirname(path.abspath(__file__))

# exp_imgs        = pd.read_csv(path.join(here, 'sampled_shape_ids.txt'), delim_whitespace=True, header=0)
image_ids       = glob.glob(path.join(PATHS['images'], "*.jpg"))
image_ids       = [path.splitext(path.basename(p))[0] for p in image_ids]
exp_imgs        = pd.DataFrame(image_ids, columns=['id'])

malignant_ids   = pd.read_csv(FILES['malignant_ids'], header=0)

shape           = pd.read_csv(FILES['cv_shape'], sep=',', header=0)
shape['id']     = [x.strip('.png') for x in shape['id']]
colour          = pd.read_csv(FILES['cv_colour'], sep=',', header=0)
colour['id']    = [x.strip('.JPG') for x in colour['id']]

data = exp_imgs\
        .merge(malignant_ids, on='id', how='inner')\
        .merge(shape, on='id', how='inner')\
        .merge(colour, on='id', how='inner')

if save_data:
    data.to_csv(FILES['cv_data'], index=False)
