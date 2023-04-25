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
    colour_data = read_data(os.path.join(paths['data'],'colours.txt'))
    # print(shape_data.head())
    colour_data["id"] = map(rm_extension, colour_data['isic_id'])
    colour_data["colours"] = map(num_colours, colour_data['identified'])
    print(colour_data.head())


if __name__ == '__main__':
    main()

