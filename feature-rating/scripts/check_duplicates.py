import cv2 as cv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from tqdm import tqdm
import concurrent.futures
import collections
from pprint import pprint
from itertools import repeat
from time import perf_counter


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


def compare_img_args(*args):
    return compare_images(*args)


def read_img(path, size, save=False):
    # img = cv.imread(path + '.JPG')
    # h, w = img.shape[:2]
    # if h > w:
    #     img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    #     h, w = img.shape[:2]
    # if np.round(w / h, 2) != np.round(4.0/3.0, 2):
    #     h = int(np.round(w * 0.75))
    #     img = img[0:h, :]
    # if w > size[0]:
    #     img = cv.resize(img, size, interpolation=cv.INTER_AREA)
    #
    if save:
        # should already be in the ISIC-dataset image directory
        save_path = os.path.join(os.getcwd(), "..", "resized")
        cv.imwrite(save_path, img)
    # img = img.astype(float) / 255

    img_result = dict(id=path, img="bob")
    return img_result


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

def main():
    img_path = os.path.join(os.getcwd(), "..", "..", "images", "ISIC-database")
    file_path = os.path.join(os.getcwd(), "..", "..", "images", "metadata")
    os.chdir(file_path)
    data = pd.read_csv("metadata.csv", usecols=["isic_id"])
    file_list = list(data["isic_id"][:10])
    size_ = (512, 384)

    # data_base = data.copy()
    # data_comparator = data.copy()
    os.chdir(img_path)

    # img_tuple = collections.namedtuple("images", [
    #     'id',
    #     'img',
    # ])

    files = ()
    start = perf_counter()
    # https://medium.com/mlearning-ai/how-do-i-make-my-for-loop-faster-multiprocessing-multithreading-in-python-8f7c3de36801
    # https://superfastpython.com/processpoolexecutor-search-text-files/
    # https://stackoverflow.com/questions/67189283/how-to-keep-the-original-order-of-input-when-using-threadpoolexecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(read_img, f, size_) for f in file_list]
        for future in concurrent.futures.as_completed(futures):
            files = files + (future.result(), )
    end = perf_counter()
    print(end-start)
    files = ()
    start = perf_counter()
    for f in file_list:
        files = files + (read_img(f, size_), )
    end = perf_counter()
    print(end-start)

    # counter = 0
    # for f in file_list:
    #     files = files + (read_img(f), )
    # for dirname, _, filename in os.walk(os.getcwd()):
    #     for f in tqdm(filename, position=0, desc="files", leave=False, colour="green", ncols=10):
    #         if f.endswith(".JPG"):
    #             files = files + (file(id=f[:-4], img=read_img(f, size=sz)),)
    #             if counter >= 20:
    #                 break
    #         counter += 1


if __name__ == '__main__':
    main()