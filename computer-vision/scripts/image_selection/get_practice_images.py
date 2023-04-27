import os
import numpy as np
import pandas as pd


def read_data(path):
    data = pd.read_csv(path, delim_whitespace=True, header=0)#sep='\t', header=0)
    return data


def get_txt(path):
    t = pd.read_csv(path, header=0)
    return t


def select_top_bottom(arr, n, pct_range):
    """Randomly select n values from the top and bottom of the array."""
    pct = int(len(arr) // pct_range)
    top_values = arr[-pct:]
    bottom_values = arr[:pct]
    top_selected = np.random.choice(top_values, n, replace=False)
    bottom_selected = np.random.choice(bottom_values, n, replace=False)
    top_indices=[]
    bottom_indices=[]
    for i in top_selected:
        top_indices.append(np.where(np.isin(arr, i))[0][0])
    for i in bottom_selected:
        bottom_indices.append(np.where(np.isin(arr, i))[0][0])

    return top_selected, top_indices, bottom_selected, bottom_indices


def main(save_data=False):
    home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')
    paths = dict(
        home=home_path,
        images=os.path.join(home_path, "images", "resized"),
        masks=os.path.join(home_path, "images", "segmented", "masks"),
        segmented=os.path.join(home_path, "images", "segmented", "images"),
        data=os.path.join(home_path, "computer-vision", "scripts", "feature-analysis")
        )
    shape_data = read_data(os.path.join(paths['data'],'shape.txt'))
    shape_data['isic_id'] = shape_data['id'].apply(lambda i: i.split('.')[0])
    duplicate_ids = get_txt('rm-list.txt')
    malignant_ids = get_txt('malignant_ids.txt')
    exp_ids = get_txt('sampled_shape_ids.txt')

    shape_data = shape_data.merge(malignant_ids, how='left', on='isic_id')
    # remove duplicates then remove the pre-selected ids
    shape_data = shape_data[~shape_data['isic_id'].isin(duplicate_ids['isic_id'])]
    shape_data = shape_data[~shape_data['isic_id'].isin(exp_ids['isic_id'])]
    
    shape_data.drop(shape_data[shape_data.compact < 0.1].index, inplace=True) 
    shape_data = shape_data.dropna().reset_index(drop=True)
    shape_data = shape_data.sort_values(by='compact', ignore_index=True).reset_index()

    y = np.array(shape_data['compact'])
    values_high, indices_high, values_low, indices_low = select_top_bottom(y, n=6, pct_range=10)

    practice_ids_low = shape_data['isic_id'][indices_low]
    practice_ids_low = practice_ids_low.sort_values()
    practice_ids_high= shape_data['isic_id'][indices_high]
    practice_ids_high = practice_ids_high.sort_values()

    if save_data:
        np.savetxt(r'practice_low_ids.txt', practice_ids_low, fmt='%s', header='isic_id', comments='')
        np.savetxt(r'practice_high_ids.txt', practice_ids_high, fmt='%s', header='isic_id', comments='')


if __name__ == '__main__':
    main(save_data=True)
