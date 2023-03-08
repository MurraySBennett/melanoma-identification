import cv2 as cv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
from pprint import pprint
from time import perf_counter
import pickle
from tqdm import tqdm


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


def compare_images(imageA, imageB, idA, idB):
    """
    :param imageA: base image
    :param imageB: comparison image
    :return: MSE [0, inf] where 0 is identical, structural similarity [-1,1] where 1 = identical), boolean comparison
    """
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, channel_axis=2)
    b = np.sum(imageA == imageB) / np.array(imageA.size)
    checks = {idA + "-" + idB: [m, s, b]}
    if m < 0.01 and s > 0.95 and b > 0.9:
        return checks
    else:
        return {}


def compare_img_args(*args):
    return compare_images(*args)


def read_img(path, size, save=False):
    img = cv.imread(path + '.JPG')
    h, w = img.shape[:2]
    if h > w:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    if np.round(w / h, 2) != np.round(4.0 / 3.0, 2):
        h = int(np.round(w * 0.75))
        img = img[0:h, :]
    if w > size[0]:
        img = cv.resize(img, size, interpolation=cv.INTER_AREA)
    if w < size[0]:
        img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)  # CUBIC is slower but better, LINEAR is the opposite
    if save:
        # should already be in the ISIC-dataset image directory
        save_path = os.path.join(os.getcwd(), "..", "resized", path + ".JPG")
        cv.imwrite(save_path, img)
    img = img.astype(float) / 255

    img_result = {path: img}
    return img_result


def read_compare(file_list, img_size):
    files = {}
    # https://medium.com/mlearning-ai/how-do-i-make-my-for-loop-faster-multiprocessing-multithreading-in-python-8f7c3de36801
    # https://superfastpython.com/processpoolexecutor-search-text-files/
    # https://stackoverflow.com/questions/67189283/how-to-keep-the-original-order-of-input-when-using-threadpoolexecutor
    start = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = [executor.submit(read_img, f, img_size, save=False) for f in file_list]
        for future in concurrent.futures.as_completed(futures):
            files.update(future.result())
    # [files.update(read_img(f, img_size, save=True)) for f in file_list]
    print(perf_counter()-start)
    files_compare = files.copy()
    comparisons = {}

    # for count_i, key_i in enumerate(tqdm(files, position=1, colour='#FFC1CC', leave=True, total=len(files))):
    for count_i, key_i in enumerate(files):
        imgA = files[key_i]
        del files_compare[key_i]
        [comparisons.update(compare_images(imgA, files_compare[imgB], key_i, imgB)) for imgB in
         files_compare.keys()]
    del files
    return comparisons


def main(save_comparisons):
    img_path = os.path.join(os.getcwd(), "..", "..", "images", "ISIC-database")
    file_path = os.path.join(os.getcwd(), "..", "..", "images", "metadata")
    os.chdir(file_path)
    data = pd.read_csv("metadata.csv", usecols=["isic_id"])
    file_list = list(data["isic_id"])

    size_ = (512, 384)

    if save_comparisons:
        comparisons = {}

        os.chdir(img_path)

        # n_chunks = 200
        # n_per_chunk = int(len(file_list) / n_chunks)

        # file_list = [file_list[i:i + n_per_chunk] for i in range(0, len(file_list), n_per_chunk)]
        # fl_copy = file_list.copy()

        # for i in tqdm(range(len(file_list)), position=0, colour='#00cdcd', leave=True, total=n_chunks):
        #     fl_copy = fl_copy[1:]
        #     compared_chunks = [read_compare(file_list[i] + fl_copy[f], size_) for f in range(len(fl_copy))]
        #     [comparisons.update(chunk) for chunk in compared_chunks]
        comparisons = read_compare(file_list, size_)
        pprint(comparisons)
        with open('comparison_data.pkl', 'wb') as f:
            os.chdir(file_path)
            pickle.dump(comparisons, f)

    else:
        with open('comparison_data.pkl', 'rb') as f:
            comparisons = pickle.load(f)


if __name__ == '__main__':
    main(save_comparisons=True)
