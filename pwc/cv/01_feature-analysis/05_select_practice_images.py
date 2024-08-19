import os
import glob
import pandas as pd
import numpy as np
# from plot_funcs import set_style

def read_shape(path):
    df = pd.read_csv(path, delim_whitespace=True, header=0)
    df['id'] = [x.strip('.png') for x in df['id']]
    return df


def read_colour(path):
    df = pd.read_csv(path, delim_whitespace=True, header=0)
    df['id'] = [x.strip('.JPG') for x in df['id']]
    return df
     

def rm_exp_images(data, exp_imgs):
    merge = data.merge(exp_imgs, on='id', how='outer', indicator=True)
    df = merge[merge['_merge']=='left_only']
    return df

 
def load_exp_images(path):
    df = pd.read_csv(path, delim_whitespace=True, header=0)
    df['id'] = [x.strip('.JPG') for x in df['id']]
    return df


def main():
    n_images = 20
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
    paths = dict(
        home=home_path,
        data=os.path.join(home_path, "feature-rating", "experiment", "melanoma-2afc", "data"),
        cv_data=os.path.join(home_path, "computer-vision", "scripts", "feature-analysis"),
        figures=os.path.join(home_path, "feature-rating", "figures"),
        mel_id=os.path.join(home_path, "computer-vision", "scripts", "image_selection"),
        )
    
    exp_images = load_exp_images(os.path.join(paths['mel_id'], 'feature-rating-image-list.txt'))
    shape_data = read_shape(os.path.join(paths['cv_data'], 'cv_shape.txt'))
    shape_data = shape_data.sort_values('id')
    colour_data = read_colour(os.path.join(paths['cv_data'], 'colours_continuous.txt'))
    colour_data = colour_data.sort_values('id')
    melanoma_ids = pd.read_csv(os.path.join(paths['mel_id'], 'malignant_ids.txt'))

    data = rm_exp_images(shape_data, exp_images)
    data = pd.merge(data, colour_data, on='id')
    data = pd.merge(data, melanoma_ids, on='id')

    # new calculations
    data['sym_combined'] = data['x_sym'] + data['y_sym']
    # this doesn't add much -- all three channels are pretty much identical.
    data['coeff_mu'] = np.round((data['coeff_1'] + data['coeff_2'] + data['coeff_3']) / 3, 3)

    # filter top and bottom 10% of data based on compactness
    data.drop(data[data.compact < 0.1].index, inplace=True)
    # reduce range of the symmetry values -- many are affected by the dermoscopy image circle.
    data.drop(data[data.sym_combined < 0.1].index, inplace=True)
    data.drop(data[data.sym_combined > 2].index,inplace=True)

    data = data.dropna().reset_index(drop=True)
    
    # summary = data.agg(
    #         {
    #             'rms': ['min','mean','median','max'],
    #             'coeff_1': ['min','mean','median','max'],
    #             'coeff_2': ['min','mean','median','max'],
    #             'coeff_3': ['min','mean','median','max'],
    #             'coeff_mu': ['min','mean','median','max']
    #         })
    # print(summary) 

    if f == 'symmetry':
        data = data.sort_values('sym_combined')
        print('High symmetrical')
        print(data.head(n_images))
        print('Low symmetry')
        print(data.tail(n_images))
    elif f == 'border':
        data = data.sort_values('compact')
        print('High border regularity')
        print(data.head(n_images))
        print('Low border regularity')
        print(data.tail(n_images))
    elif f == 'colour':
        # colour -- when you have it
        # print(data.describe())
        # 25%  ~17% and 75% ~ 31
        data.drop(data[data.rms < 15].index, inplace=True)
        data.drop(data[data.rms > 30].index, inplace=True)
        data = data.sort_values('rms')
        print(data.head(n_images))
        print(data.tail(n_images))

    # colours = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
    # set_style(colour_list=colours, fontsize=14)
    return data
    
if __name__ == '__main__':
    feature = ['symmetry', 'border', 'colour']
    for f in feature:
        data = main()