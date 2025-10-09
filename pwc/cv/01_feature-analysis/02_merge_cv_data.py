import sys
from os import path
import pandas as pd

save_data = True
here = path.dirname(path.abspath(__file__))

# exp_imgs        = pd.read_csv(path.join(here, 'sampled_shape_ids.txt'), delim_whitespace=True, header=0)
image_ids       = glob.glob(os.path.join(PATHS['images'], "*.jpg"))
image_ids       = [os.path.splitext(os.path.basename(p))[0] for p in exp_imgs]
exp_imgs        = pd.DataFrame(image_ids, columns=['id'])
exp_imgs       = [x.strip('jpg')[0] for x in exp_imgs['id']]

malignant_ids   = pd.read_csv(FILES['malignant_ids'], header=0)

shape           = pd.read_csv(FILES[cv_shape], delim_whitespace=True, header=0)
shape['id']     = [x.strip('.png') for x in shape['id']]
colour          = pd.read_csv(FILES[cv_colour], delim_whitespace=True, header=0)
colour['id']    = [x.strip('.JPG') for x in colour['id']]

data = exp_imgs\
        .merge(malignant_ids, on='id', how='inner')\
        .merge(shape, on='id', how='inner')\
        .merge(colour, on='id', how='inner')

if save_data:
    data.to_csv(FILES['cv_data'], index=False)
