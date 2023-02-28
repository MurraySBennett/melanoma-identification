from __future__ import division
import os
import cv2
import numpy as np
from time import time, perf_counter
import concurrent.futures


## Parallel Processing
## https://www.machinelearningplus.com/python/parallel-processing-python/

def get_files(path, n_files):
    f = []
    names = []
    roots = []
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".JPG") or name.endswith(".jpg"):
                names.append(name)
                f.append(os.path.join(root, name))
                roots.append(root)
            if n_files is not None and count == n_files:
                break
            count += 1
    return f, names


def img_process_args(args):
    return process_img(*args)


def process_img(file_name):
    size = (512, 384)  # 256,192
    save_path = os.path.join(os.getcwd(), "images", "resized")
    img_path = os.path.join(os.getcwd(), "images", "ISIC-2020")

    img = cv2.imread(img_path + "\\" + file_name)
    h, w = img.shape[:2]

    if h > w:
        # set width as longest side
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    if np.round(w / h, 2) != np.round(4.0/3.0, 2):
        # set aspect to 4:3 by setting the height to be 0.75*width
        h = int(np.round(w * 0.75))
        img = img[0:h, :]  # img[rows,columns]

    # resize images for memory -- I think they'll all be resized down to 256,192
    # for the experiment and the image analysis.
    if w > size[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    if save_path is not None:
        save_file = save_path +"\\" + file_name
        cv2.imwrite(save_file, img)
        print(save_file)
        return save_file
    else:
        return "processed file_name"


if __name__ == '__main__':
    start = perf_counter()
    n_files = None

    # until you figure out how to run funcitons with multiple arguments, or to just use global variables properly,
    # you will need to update the img_path and the img size variables INSIDE the process_img function
    img_path = os.path.join(os.getcwd(), "images")
    f, names = get_files(os.path.join(img_path, "ISIC-2020"), n_files)
    print(names)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_img, names))  # target_size, file, save_path

    end = perf_counter()
    # for result in results:
    #     print(result)
    print(f"total runtime for all images = {end - start}s")

# for i in range(len(f)):
    # img = cv2.imread(f[i])
    # h, w = img.shape[:2]
    # # print(f"Initial shape (w,h): {w, h}, final shape: {target_size}")
    #
    # if h > w:
    #     # set width as longest side
    #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #     h, w = img.shape[:2]
    #
    # if np.round(w / h, 2) != np.round(4.0/3.0, 2):
    #     # set aspect to 4:3 by setting the height to be 0.75*width
    #     h = int(np.round(w * 0.75))
    #     img = img[0:h, :]  # img[rows,columns]
    #
    # # resize images for memory -- I think they'll all be resized down to 256,192
    # # for the experiment and the image analysis.
    # if w > target_size[0]:
    #     img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    #
    # # cv2.imshow("Image", img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    #
    # images.append(img)
    # # size.append([h, w])
    # size_w.append(w)
    # size_h.append(h)


# n_bins = 20
# fig, axs = plt.subplots(1, 2, sharex=True, tight_layout=True)
# # We can set the number of bins with the *bins* keyword argument.
# axs[0].hist(size_w, bins=n_bins)
# axs[1].hist(size_h, bins=n_bins)
# plt.show()
