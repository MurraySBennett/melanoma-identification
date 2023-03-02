import cv2 as cv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import concurrent.futures
import collections
import pprint


def show_images(img1, img2):
    f, ax = plt.subplots(1, 2, figsize=(5, 5))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[0].axis('off')
    ax[1].axis('off')
    f.tight_layout()
    plt.show()


def mse(img1, img2):
    """
    :param img1: base image
    :param img2: comparison image
    :return: mean square error -- 0 indicates perfect similarity
    """
    err = np.sum(np.subtract(img1, img2) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err


def compare_images(imageA, imageB):
    """
    :param imageA: base image
    :param imageB: comparison image
    :return: MSE [0, inf] where 0 is identical, structural similarity [-1,1] where 1 = identical), boolean comparison
    """
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, channel_axis=2)
    b = np.sum(imageA == imageB) == imageA.size
    return {'mse': m, 'ssim': s, 'bool': b}


def compare_img_args(args):
    return compare_images(*args)


def read_img(path, size):
    img = cv.imread(path + '.JPG')
    img = cv.resize(img, size, interpolation=cv.INTER_AREA)
    img = img.astype(float) / 255
    result = {'id': path, 'img': img}
    print("I just processed image", path)
    return result


def read_img_args(kwargs):
    return read_img(**kwargs)

# for count_i, value_i in enumerate(tqdm(data_base["isic_id"], position=0, desc="i", leave=False, colour='green', ncols=70)):
#     image1 = read_img(data_base["isic_id"][count_i] + ".JPG", size)
#     data_comparator = data_comparator.drop(index=0).reset_index(drop=True)
#     for count_j, value_j in enumerate(tqdm(data_comparator["isic_id"], position=1, desc="j", leave=False, colour="red", ncols=70)):
#         image2 = read_img(data_comparator["isic_id"][count_j] + ".JPG", size)
#         m, s, b = compare_images(image1, image2)
#         if b:
#             show_images(image1, image2)
#             print(value_i, "=", value_j)
#             print(f'mse = {m}, ss = {s}, boolean comparator = {b}')
#             break
#
#     if b is not None and b:
#         break

# for dirname, _, filenames in os.walk(os.getcwd()):
#     for filename in filenames:
#         if filename == "ISIC_0079038.JPG":
#             image1 = read_img(filename, size)
#         if filename == "ISIC_8521950.JPG":
#             image2 = read_img(filename, size)
#
# m, s, b = compare_images(image1, image2)
# print(f'mse = {m}, ss = {s}, boolean comparator = {b}')
# show_images(image1, image2)

if __name__ == '__main__':
    img_path = os.path.join(os.getcwd(), "..", "..", "images", "ISIC-database")
    file_path = os.path.join(os.getcwd(), "..", "..", "images", "metadata")
    os.chdir(file_path)
    SIZE_ = (512, 384)
    data = pd.read_csv("metadata.csv", usecols=["isic_id"])
    file_list = list(data["isic_id"][:10])
    print(file_list)

    # data_base = data.copy()
    # data_comparator = data.copy()
    os.chdir(img_path)

    file = collections.namedtuple("images", [
        'id',
        'img',
    ])

    files = ()
    counter = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # result = executor.map(read_img, file_list)
        result = executor.map(read_img_args, {"size": SIZE_, "path": file_list})

    pprint(tuple(result))

# for dirname, _, filename in os.walk(os.getcwd()):
#     for f in tqdm(filename, position=0, desc="files", leave=False, colour="green", ncols=10):
#         if f.endswith(".JPG"):
#             files = files + (file(id=f[:-4], img=read_img(f, size=sz)),)
#             if counter >= 20:
#                 break
#         counter += 1
# print(files)

# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for counter, i in enumerate(tqdm(data_base['isic_id'][counter], colour='green', ncols=70, desc="base progress")):
#         results = list(executor.map(process_img, names))  # target_size, file, save_path

# for result in results:
#     print(result)
