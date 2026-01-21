import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from .file_management import save_img, set_logger
from .minimise_dca import *

def crop_dca(img, black_val=15):
    """Detects dark corner areaas and returns a cropped ROI """
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray[0,:]  = 0
    gray[-1,:] = 0
    gray[:, 0] = 0
    gray[:,-1] = 0

    _, mask = cv.threshold(gray, black_val, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)    

    if not contours:
        return img, (0, 0, w, h), False
    largest = max(contours, key=cv.contourArea)
    lens_mask = np.zeros((h, w), dtype=np.uint8)
    cv.drawContours(lens_mask, [largest], -1, 255, -1)
    img_cleaned = cv.bitwise_and(img, img, mask=lens_mask)
    x, y, fw, fh = cv.boundingRect(largest)
    if fw < w * 0.5:
        return img, (0, 0, w, h), False
    cropped = img_cleaned[y:y+fh, x:x+fw]
    return cropped, (x, y, fw, fh), True

    # buffer = 10
    # corners = [
    #     gray[0:buffer, 0:buffer],         # Top-Left
    #     gray[0:buffer, w-buffer:w],       # Top-Right
    #     gray[h-buffer:h, 1:buffer],       # Bottom-Left
    #     gray[h-buffer:h, w-buffer:w]      # Bottom-Right
    # ]
    # corner_avg = np.mean([np.mean(c) for c in corners])
    # if corner_avg > black_val:
    #     return img, (0, 0, w, h), False
    # _, mask = cv.threshold(gray, black_val, 255, cv.THRESH_BINARY)
    # contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     return img, (0, 0, w, h), False
    # largest = max(contours, key=cv.contourArea)
    # x, y, fw, fh = cv.boundingRect(largest)
    # if fw < w * 0.5: # if focal area is less than 50% width, probably noise (hopefullY?)
    #     return img, (0, 0, w, h), False
    # cropped = img[y:y+fh, x:x+fw]
    # return cropped, (x, y, fw, fh), True


def process_img(img, n_clusters=6, segment='otsu', save_img_label=None):
    """ combine all functions for full processing method """
    # logger = set_logger()
    try:
        original_h, original_w = img.shape[:2]
        original = img.copy()

        img_cropped, roi, was_cropped = crop_dca(img)
        x, y, w_roi, h_roi = roi
        
        # dca = get_dca(img)
        # img = paint_dca(dca[0], dca[1], method='inpaint_ns')

        img_proc = hair_rm(img_cropped)
        img_proc = colour_cluster(img_proc, n_clusters)
        img_proc = my_clahe(img_proc)

        if segment == 'otsu':
            mask_cropped = otsu_segment(img_cropped, img_proc)
        elif segment == 'grabcut':
            mask_cropped = grabcut_segment(img_cropped, img_proc)
        full_mask = np.zeros((original_h, original_w), dtype=np.uint8)        
        full_mask[y:y+h_roi, x:x+w_roi] = mask_cropped

        if save_img_label is not None:
            save_img_label(full_mask, save_img_label) 
        return full_mask
    except Exception as e:
        print(f"{img.stem} error: {e}")


def display(img, labels):
    """ show images """
    plt.figure(figsize=(len(img*2), 3))
    for i in range(len(img)):
        plt.subplot(1, len(img), i+1)
        plt.imshow(img[i])
        plt.axis('off')
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()


def hair_rm(img):
    """ reduce hair and skin texture artefacts """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Kernel for the morphological filtering
    kernel = cv.getStructuringElement(1, (5, 5))
    # Perform blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    closing = cv.morphologyEx(blackhat, cv.MORPH_CLOSE, kernel)
    bhg = cv.GaussianBlur(closing, (3, 3), cv.BORDER_DEFAULT)
    # intensify the hair countours in preparation for the inpainting algorithm
    ret, thresh2 = cv.threshold(bhg, 1, 255, cv.THRESH_BINARY)
    # inpaint the original image depending on the mask
    processed  = cv.inpaint(img, thresh2, 6, cv.INPAINT_TELEA)
    return processed 


def colour_cluster(img, n_clusters):
    """ cluster image colours for improved border detection """
    median = cv.medianBlur(img, 5)
    Z = np.float32(median.reshape((-1, 3)))
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # random_centers is stochastic, so slight variability will exist
    _, label, center = cv.kmeans(Z, n_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


def my_clahe(img):
    """ contrast limited adaptive histogram equalisation.
    returns enhanced image """
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = clahe.apply(h)
    s = clahe.apply(s)
    v = clahe.apply(v)
    enhanced_hsv = cv.merge((h, s, v))
    return cv.cvtColor(enhanced_hsv, cv.COLOR_LAB2BGR)
    
    # lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # l, a, b = cv.split(lab)
    # l = clahe.apply(l)
    # l_enhanced = cv.merge((l, a, b))
    # enhanced = cv.cvtColor(l_enhanced, cv.COLOR_LAB2BGR)
    # return enhanced


def otsu_segment(img, enhanced):
    """ otsu thresholding """
    canny_otsu = otsu_thresh(enhanced)
    largest_contour = get_contours(canny_otsu)
    otsu_mask = contour_mask(img, largest_contour)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    otsu_mask[gray_img == 0] = 0
    return otsu_mask

def otsu_thresh(img):
    """ threshold image using otsu method and return canny edge detection """
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_img)
    converted_img = cv.GaussianBlur(S + V, (5, 5), 0)

    valid_pixels = converted_img[converted_img > 0]
    if len(valid_pixels) > 0:
        thresh_val, _ = cv.threshold(valid_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, otsu = cv.threshold(converted_img, thresh_val, 255, cv.THRESH_BINARY)
    else:
        otsu=np.zeros_like(converted_img)

    # blur = cv.GaussianBlur(converted_img, (5,5), 0)
    # ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    canny_otsu = cv.Canny(otsu, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    canny_otsu = cv.dilate(canny_otsu, kernel, iterations=1)
    return canny_otsu


def get_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # if len(contours) != 0:
    #     largest_contour = max(contours, key=cv.contourArea)
    # return largest_contour    
    if not contours:
        return None
    return max(contours, key=cv.contourArea)


def contour_mask(img, contour):
    h, w, d = img.shape
    # mask = np.zeros((h, w)).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contour is not None:
        cv.fillPoly(mask, pts=[contour], color=255)
    # cv.fillPoly(mask, pts=[contour], color=(255, 255, 255))
    return mask


def green_mask(img):
    """ mask according to a green-value threshold """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([100, 255, 255])
    mask_g = cv.inRange(hsv, lower_green, upper_green)

    ret, inv_mask = cv.threshold(mask_g, 127, 255, cv.THRESH_BINARY_INV)
    # res = cv.bitwise_and(img, img, mask=mask_g)
    return inv_mask


def segment(img):
    contoured_image = img.copy()  # copy() so that you are not drawing over the original image
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_img)
    converted_img = S + V
    blur = cv.GaussianBlur(converted_img, (5, 5), 0)

    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    canny_h = cv.Canny(H, 50, 120)
    canny_s = cv.Canny(S, 100, 200)
    canny_v = cv.Canny(V, 50, 100)
    canny_otsu = cv.Canny(otsu, 100, 200)
    
    processed = canny_otsu
    # Dilate img to 'close' spaces -- connect canny edges
    kernel = np.ones((3, 3), np.uint8)
    processed = cv.dilate(processed, kernel, iterations=1)

    contours, hierarchy = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #  check contours
    w,h,d = img.shape
    new_contours = []
    
    # find largest contour and assume that it's the correct one
    cntsSorted = sorted(new_contours, key=lambda x: cv.contourArea(x))
    
    if len(contours) != 0:
        largest_contour = max(contours, key=cv.contourArea)
    
    cv.drawContours(contoured_image, [largest_contour], 0, (255, 0, 0), 1)

    mask = np.zeros((w, h)).astype(np.uint8)
    cv.fillPoly(mask, pts=[largest_contour], color=(255, 255, 255))

    return mask, largest_contour, contoured_image



def grabcut(img, enhanced_img, inv_mask):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # inv_mask is 384x512 of 0s or 255. 
    mask_threshold = mask.shape[0] * inv_mask.shape[1] * 255 * 0.75
    if (np.sum(inv_mask[:]) < mask_threshold):
        new_mask = inv_mask
        mask[new_mask == 0] = 0
        mask[new_mask == 255] = 1
        dim = cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        grabcut_img = img * mask2[:, :, np.newaxis]
    else: #### GRABCUT
    # initialise rectangle based on image dimensions
        s = (img.shape[0] / 10, img.shape[1] / 10)
        rect = (int(s[0]), int(s[1]), int(img.shape[0] - (3/10) * s[0]), int(img.shape[1] - s[1]))
        dim = cv.grabCut(enhanced_img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        grabcut_img = img*mask2[:, :, np.newaxis]

    return grabcut_img


def grabcut_segment(img, enhanced):
    # github.com/fitushar/skin-lesion-segmentation...

    ###### mask generation
    inv_mask = green_mask(enhanced)
    grabcut_img = grabcut(img, enhanced, inv_mask)
    ###### binarisation
    img_mask = cv.medianBlur(grabcut_img, 5)
    ret, segmented_mask = cv.threshold(img_mask, 0, 255, cv.THRESH_BINARY)
    
    return segmented_mask
