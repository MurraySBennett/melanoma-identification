import os
import glob
import cv2 
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_anns(anns, img):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        print(ann['area'], ann['bbox'], check_perimeter(img, ann['bbox'], border=10))
        if ann['area'] < 150:
            continue
        if check_perimeter(img, ann['bbox'], border = 10) == False:
            continue

        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def check_perimeter(image, bbox, border=5):
    """Check if the bounding box is within a border distance of the image perimeter."""
    height, width = image.shape[:2]
    perimeter = np.array([border, border, width - 1 - border, height - 1 - border])
    bbox = np.asarray(bbox)
    return np.all((bbox[:2] >= perimeter[:2]) & (bbox[2:] <= perimeter[2:]))
    # height, width = image.shape[:2]
    # perimeter = np.array([border, border, width - 1 - border, height - 1 - border])
    # print(height, width, perimeter)
    # return np.all(np.logical_or(bbox <= perimeter[:2], bbox >= perimeter[2:]))



n_images = 1
home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')
paths = dict(
    home=home_path,
    images=os.path.join(home_path, "images", "resized"),
    masks=os.path.join(home_path, "images", "segmented", "sam-masks"),
    segmented=os.path.join(home_path, "images", "segmented", "images"),
    data=os.path.join(home_path, "images", "metadata"),
    checkpoint=os.path.join(home_path, "computer-vision", "sam", "model-checkpoint")
    )

image_paths = glob.glob(os.path.join(paths['images'], '*.JPG'))
image_paths = sorted(image_paths)
if n_images is not None:
    image_paths = image_paths[:n_images]
mask_paths = glob.glob(os.path.join(paths['masks'], '*.png'))
mask_paths = sorted(mask_paths)

checkpoint=glob.glob(os.path.join(paths['checkpoint'], '*'))
print(checkpoint)


def main():
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint[0])
    mask_generator = SamAutomaticMaskGenerator(sam)

    image = cv2.imread(image_paths[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(5,5))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    masks = mask_generator.generate(image)
    # print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    show_anns(masks, image)
    plt.axis('off')
    plt.show()


if __name__=='__main__':
    main()
