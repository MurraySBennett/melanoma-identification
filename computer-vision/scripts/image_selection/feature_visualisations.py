import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


def read_data(path):
    data = pd.read_csv(path, sep='\t', header=0)
    return data


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


def sample_data(y, n):
    loc = np.mean(y)
    sd = np.std(y)
    sample_indices = []
    while len(sample_indices) < n:
        tmp_n = n - len(sample_indices)
        tmp_indices = np.round(np.random.normal(loc=loc, scale=sd, size=tmp_n)).astype(int)
        tmp_indices = np.clip(tmp_indices, 0, len(y)-1)
        sample_indices = np.append(sample_indices, tmp_indices)
        sample_indices = np.unique(sample_indices)
        # print(f'sample index length: {len(sample_indices)}, n to be added: {tmp_n}, {sample_indices}, {tmp_indices}')
    sample_values = y[sample_indices]
    return sample_values, sample_indices


def main():
    home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')
    paths = dict(
        home=home_path,
        images=os.path.join(home_path, "images", "resized"),
        masks=os.path.join(home_path, "images", "segmented", "masks"),
        segmented=os.path.join(home_path, "images", "segmented", "images"),
        data=os.path.join(home_path, "computer-vision", "scripts", "feature-analysis")
        )
    # shape_data = read_data(os.path.join(paths['data'],'shape.txt'))
    # colour_data = read_data(os.path.join(paths['data'],'colours.txt'))
    # print(shape_data.head())
    simdata_len = 1000
    max_value = 1000
    shape_data = pd.DataFrame(np.random.randint(0,max_value,size=(simdata_len, 4)), columns=["id","x_sym","y_sym","compact"])
    shape_data["id"] = np.arange(1,simdata_len+1,1)
    shape_data = shape_data.sort_values(by='x_sym', ignore_index=True).reset_index()
    shape_data["sorted_id"] = np.arange(1,simdata_len+1,1)
    # colour_data["id"] = map(rm_extension, colour_data['isic_id'])
    # colour_data["colours"] = map(num_colours, colour_data['identified'])
    # print(shape_data.head())
    
    x = shape_data['sorted_id']
    y = shape_data['x_sym']

    # normal random sample from sorted data
    # loc_y = np.mean(y)
    # sd_y = np.std(y)
    # n_samples = int(simdata_len * 0.2)
    # sample_indices = np.round(np.random.normal(loc=loc_y, scale=sd_y, size=n_samples)).astype(int)
    # sample_indices = np.clip(sample_indices, 0, len(x)-1)
    # sample_values = y[sample_indices]
    sample_values, sample_indices = sample_data(y, n=int(simdata_len*0.2))
    original_indices=x[sample_indices]

    ## plot shape data
    fig, ax = plt.subplots(1, 3, figsize=(8,5))
    ax[0].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), lw=5, c='#f85ed6', zorder=1)
    ax[0].scatter(x, y, c='#6721ff', alpha=0.9, s=100, zorder=2)
    ax[0].scatter(original_indices, sample_values, c='#f79729', s=50, zorder=3)
    # ax.hlines(loc_y, xmin=1, xmax=simdata_len,lw=2)
    # ax.fill_between(x, loc_y+sd_y, loc_y-sd_y, facecolor='#FA5F55',alpha=0.5)
    ax[0].set_xlabel('order')
    ax[0].set_ylabel('x symmetry')
    ax[0].set_title('shape symmetry')
    # ax.axis('off')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_aspect('equal','box')

    ## histogram the selected values to see if you have a normal distribution
    bins = range(0, max_value, max_value//10)
    ax[1].hist(y, bins=bins)
    ax[1].set_title('raw values')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    # ax[1].set_aspect('equal','box')
    ax[1].set_adjustable('box')

    ax[2].hist(sample_values, bins=bins)
    ax[2].set_title('sampled values')
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    # ax[2].set_aspect('equal','box')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

