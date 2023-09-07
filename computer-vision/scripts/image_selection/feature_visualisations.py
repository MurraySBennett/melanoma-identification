import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


def read_data(path):
    return pd.read_csv(path, delim_whitespace=True, header=0)#sep='\t', header=0)

def get_txt(path):
    return pd.read_csv(path, header=0)

def show_images(images):
    f, ax = plt.subplots(1, len(images), figsize=(len(images)*1.5, 3))
    for i, img in enumerate(images):
        ax[i].imshow(img)
        ax[i].axis('off')
        f.tight_layout()
    plt.show()

def num_colours(colours):
    """ to be used inside apply ?? """
    colours = colours.strip('[]')
    n_colours = 0 if colours == ''  else len(colours.split(','))
    return n_colours

def rm_extension(file_name):
    no_extension = file_name.split('.')[0]
    return no_extenstion

def sample_data(y, n, sample_indices):
    loc = np.mean(y)
    sd = np.std(y)
    # sample_indices = []
    while len(sample_indices) < n:
        tmp_n = n - len(sample_indices)
        tmp_indices = np.round(np.random.normal(loc=loc, scale=sd, size=tmp_n)*len(y)).astype(int)
        tmp_indices = np.clip(tmp_indices, 0, len(y)-1)
        sample_indices = np.append(sample_indices, tmp_indices)
        sample_indices = np.unique(sample_indices.astype(int))
        # print(f'{len(sample_indices)//n}%')
        # print(f'sample index length: {len(sample_indices)}, n to be added: {tmp_n}, {sample_indices}, {tmp_indices}')
    sample_values = []
    for i in sample_indices:
        sample_values.append(y[i])
    return sample_values, sample_indices

def main(target_n=100, save_data=False):
    home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')
    paths = dict(
        home=home_path,
        images=os.path.join(home_path, "images", "resized"),
        masks=os.path.join(home_path, "images", "segmented", "masks"),
        segmented=os.path.join(home_path, "images", "segmented", "images"),
        data=os.path.join(home_path, "computer-vision", "scripts", "feature-analysis")
        )
    #colour_data = read_data(os.path.join(paths['data'],'colours.txt'))
    shape_data = read_data(os.path.join(paths['data'],'shape.txt'))
    shape_data['isic_id'] = shape_data['id'].apply(lambda i: i.split('.')[0])
    duplicate_ids = get_txt('rm-list.txt')
    malignant_ids = get_txt('malignant_ids.txt')

    shape_data = shape_data.merge(malignant_ids, how='left', on='isic_id')
    shape_data = shape_data[~shape_data['isic_id'].isin(duplicate_ids['isic_id'])]

    shape_data.drop(shape_data[shape_data.compact < 0.1].index, inplace=True) 
    shape_data = shape_data.dropna().reset_index(drop=True)
    shape_data = shape_data.sort_values(by='compact', ignore_index=True).reset_index()


    max_value = np.max(shape_data['compact'])
    n_images = shape_data.shape[0]
    shape_data["sorted_id"] = np.arange(1,n_images+1,1)


    x = np.array(shape_data['sorted_id'])
    y = np.array(shape_data['compact'])
    melanoma_idx = np.where(shape_data['malignant']==1)[0]

    sample_values, sample_indices = sample_data(y, n=target_n, sample_indices=melanoma_idx)
    sampled_ids = shape_data['isic_id'][sample_indices]
    sampled_ids = sampled_ids.sort_values()
    sampled_ids = [i+'.JPG' for i in sampled_ids['isic_id']]

    if save_data:
        np.savetxt(r'feature-rating-image-list.txt', sampled_ids, fmt='%s', header='id', comments='')
    
    original_indices = []
    for i in sample_indices:
        original_indices.append(x[i])

    melanoma_vals=[]
    for i in melanoma_idx:
        melanoma_vals.append(y[i])

    ## plot shape data
    fig, ax = plt.subplots(1, 3, figsize=(8,5))
    ax[0].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), lw=5, c='#f85ed6', zorder=1)
    ax[0].scatter(x, y, c='#6721ff', alpha=0.9, s=100, zorder=2)
    ax[0].scatter(original_indices, sample_values, c='#f79729', s=50, zorder=3)
    ax[0].scatter(melanoma_idx, melanoma_vals, c='black', s=30, alpha=0.8, zorder=3.5)
    # ax.hlines(loc_y, xmin=1, xmax=simdata_len,lw=2)
    # ax.fill_between(x, loc_y+sd_y, loc_y-sd_y, facecolor='#FA5F55',alpha=0.5)
    ax[0].set_ylim(0,1)
    ax[0].set_xlabel('order')
    ax[0].set_ylabel('x symmetry')
    ax[0].set_title('shape symmetry')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    # ax[0].set_aspect('equal','box')

    ## histogram the selected values to see if you have a normal distribution
    nbins = 10
    bins = np.round(np.linspace(0, max_value, 10),2)

    ax[1].hist(y, bins=bins)
    ax[1].set_title('raw values')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    # ax[1].set_aspect('equal','box')
    ax[1].set_adjustable('box')
    
    ax[2].hist(sample_values, bins=bins, label='all')
    ax[2].hist(melanoma_vals, bins=bins, label='melanoma')
    ax[2].set_title('sampled values')
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].legend()
    # ax[2].set_aspect('equal','box')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 10_000 images is roughly 40,500 trials, or 80 participants completing 500 trials -- see melanoma-identification/feature-rating/btl-simulation/scripts/run.py
    target_n = 10000
    save_data = False # you really should have set a random seed for this.
    main(target_n, save_data)

