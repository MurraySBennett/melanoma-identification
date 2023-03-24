import numpy as np
import cv2 as cv

# remove white perimeter
def trim_border(img, percent):
    height, width, channel = img.shape
    trim_width = int(width * percent / 100)
    trim_height = int(height * percent / 100)
    trimmed_img = img[trim_height:height-trim_height, trim_width:width-trim_width]
    return trimmed_img

# find dark corner areas and reduce the image size to minimze areas
def calculate_reduction_rate(center, radius, width, height):
    """
    Calculate the amount of pixels to remove from the horizontal and vertical
    Returns
    -------
    n pixels to remove, top edge lim, bottom edge limit, left edge limit, right edge limit borders of an image
    """
    l_edge = int(center[0] - radius) if center[0] - radius >= 0 else 0
    r_edge = int(width - (center[0] + radius)) if center[0] + radius < width else 0
    t_edge = int(center[1] - radius) if center[1] - radius >= 0 else 0
    b_edge = int(height - (center[1] + radius)) if center[1] + radius < height else 0

    # calculate the vert and horizontal totals
    vertical = t_edge + b_edge
    horizontal = l_edge + r_edge

    # take the smallest value of the 2, this is the maximum we can remove to keep the image square
    r = min([vertical, horizontal])
    return r, t_edge, b_edge, l_edge, r_edge, vertical, horizontal


def get_dca(image):
    """
    The intention is to run this process prior to rezising images
    """
    img = image.copy()
    height, width = img.shape[:2]

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_thresh = 100
    ret, thresh_img = cv.threshold(img, img_thresh, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        big_contour = max(contours, key=cv.contourArea)

    (x,y), radius = cv.minEnclosingCircle(np.asarray(big_contour))
    center = (int(x), int(y))
    radius = int(radius) - 2

    mask  = cv.circle(np.ones(img.shape), center, radius,(0,255,0),-1)
    mask[mask==1] = 255
    mask = mask // 255

    # reduce image size to minimise dca
    r, top, bottom, left, right, vertical, horizontal = calculate_reduction_rate(center, radius, width, height)
    if r != 0:
        cropped_img = np.copy(image[bottom:height-top, left:width-right])
        cropped_mask = np.copy(image[bottom:height-top, left:width-right])
    else:
        cropped_mask = mask
        cropped_img = np.copy(image)

    return (cropped_img, cropped_mask)


def paint_dca(image, mask, method='inpaint_ns'):
    """Remove DCA from a specified image. 
    Removal methods:
        inpaint_ns = Navier Stokes method
        inpaint_telea = Telea method
    
    Parameters
    ----------
    image
        image to remove DCA from
    mask
        mask of DCA to remove
    removal_method
        inpainting method to use The default is 'inpaint_ns'.
    Returns
    -------
    inpainted_image
        final inpainted image
    """
    if removal_method == 'inpaint_ns':
        flags = cv.INPAINT_NS
    elif removal_method == 'inpaint_telea':
        flags = cv.INPAINT_TELEA
    else:
        pass
    
    inpainted_image = cv.inpaint(image, mask.astype(np.uint8), inpaintRadius=10, flags=flags)

    return inpainted_image

