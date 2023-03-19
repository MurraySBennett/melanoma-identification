import pandas as pd
import numpy as np
import cv2 as cv
import os, os.path
import matplotlib.pyplot as plt

def calculate_reduction_rate(center, radius):
    """
    Calculate the amount of pixels to remove from the horizontal and vertical
    Returns
    -------
    n pixels to remove, top edge lim, bottom edge limit, left edge limit, right edge limit borders of an image
    """
    l_edge = int(center[0] - radius) if center[0] - radius >= 0 else 0
    r_edge = int(512 - (center[0] + radius)) if center[0] + radius < 512 else 0
    t_edge = int(center[1] - radius) if center[1] - radius >= 0 else 0
    b_edge = int(384 - (center[1] + radius)) if center[1] + radius < 384 else 0

    # calculate the vert and horizontal totals
    vertical = t_edge + b_edge
    horizontal = l_edge + r_edge

    # take the smallest value of the 2, this is the maximum we can remove to keep the image square
    r = min([vertical, horizontal])

    return r, t_edge, b_edge, l_edge, r_edge


def get_dca(image):
    img = image.copy()
    
    img[0,:] = img[1,:]
    img[-1,:] = img[-2,:]
    img[:,0] = img[:,1]
    img[:,-1] = img[:,-2]


    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_thresh = 100
    ret, thresh_img = cv.threshold(img, img_thresh, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contoursB = []
    big_contour = []
    maxArea = 0
    # for i in contours:
    #     area = cv.contourArea(i) #--- find the contour having biggest area ---
    #     if(area > maxArea):
    #         maxArea = area
    #         big_contour = i 
    #         contoursB.append(i)
    
    if len(contours) != 0:
        big_contour = max(contours, key=cv.contourArea)

    (x,y), radius = cv.minEnclosingCircle(np.asarray(big_contour))
    center = (int(x), int(y))
    radius = int(radius) - 2

    mask  = cv.circle(np.ones(img.shape), center, radius,(0,255,0),-1)

    mask[mask==1] = 255
    mask = mask // 255

    # reduce image size to minimise dca
    r, top, bottom, left, right = calculate_reduction_rate(center, radius)
    if r != 0:
        vert_r = r
        horz_r = r
        new_top = top if top <= vert_r else vert_r
        vert_r -= new_top
        new_bottom = 384 - vert_r
        new_left = left if left <= horz_r else horz_r
        horz_r -= new_left
        new_right = 512 - horz_r
        cropped_mask = np.copy(mask[new_top:new_bottom, new_left:new_right])
        cropped_img = np.copy(image[new_top:new_bottom, new_left:new_right])
    else:
        cropped_mask = np.copy(mask)
        cropped_img = np.copy(image)

    # plt.figure(figsize=(5,5))
    # plt.imshow((cv.drawContours(img, contours, -1, (255,255,0), 3)).astype(np.uint8))
    # plt.imshow(cropped_mask, cmap="gray")
    # plt.show()
    # return mask.astype(np.uint8)
    return (cropped_img, cropped_mask)


# def reduce_intensity(image, mask):
#     """Reduce the intensity of the DCA by removing as much of the surrounding border 
#     as possible. This method calculates the total horizontal and vertical distances
#     and uses the minima to retain a square image.
#     Parameters
#     ----------
#     image : np.ndarray
#         image to crop
#     mask : np.ndarray
#         corresponding mask to crop
#     Returns
#     -------
#     np.ndarray
#         cropped image
#     np.ndarray
#         cropped mask
#     """
#     # Convert the image to greyscale
#     gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

#     # Threshold the image, same as done for masking process
#     image_thresh = 100
#     ret, thresh = cv.threshold(gray, image_thresh, 255, cv.THRESH_BINARY)

#     # Retrieve all of the contours
#     contours, hierarchy = cv.findContours(thresh, 1, 2)

#     # Find the largest contour
#     contours_b = []
#     big_contour = []
#     max = 0

#     for i in contours:
#         area = cv.contourArea(i)
#         if max < area:
#             max = area
#             big_contour = i
#             contours_b.append(i)

#     # Find the minimum enclosing circle center coordinates and radius
#     (x, y), radius = cv.minEnclosingCircle(big_contour)
#     center = (int(x), int(y))
#     radius = int(radius) - 2

#     r, t_edge, b_edge, l_edge, r_edge = calculate_reduction_rate(center, radius)

#     if r != 0:
#         # Only go through cropping process if there is border to be removed

#         # Calculate how much to actually remove from the image
#         # How many pixels left to remove?
#         vertical_r = r
#         horizontal_r = r 
 
#         new_top = t_edge if t_edge <= vertical_r else vertical_r
#         vertical_r -= new_top
#         new_bottom = 224 - vertical_r
        
#         new_left = l_edge if l_edge <= horizontal_r else horizontal_r
#         horizontal_r -= new_left
#         new_right = 224 - horizontal_r
        
#         cropped_mask = np.copy(mask[new_top:new_bottom, new_left:new_right])
#         cropped_image = np.copy(image[new_top:new_bottom, new_left:new_right])
#     else:
#         cropped_mask = np.copy(mask)
#         cropped_image = np.copy(image)

#     return cropped_image, cropped_mask


def run_super_resolution(image, mask):
    """Enhance the resolution of the image to combat the reduction in quality.
    Parameters
    ----------
    image : np.ndarray
        the image to modify
    mask : np.ndarray
        the corresponding mask for the image
    Returns
    -------
    np.ndarray
        the modified image
    np.ndarray
        the modified mask
    """
    print(image.shape)
    super_res = cv.dnn_superres.DnnSuperResImpl_create()
    path = r'./Models/EDSR_x4.pb'
    super_res.readModel(path)
    super_res.setModel("edsr", 4)
    upsampled = super_res.upsample(image)
    upsampled = cv.resize(upsampled,dsize=(512,384))
    
    mask = mask.astype(np.uint8)
    upsampled_mask = super_res.upsample(cv.cvtColor(mask, cv.COLOR_GRAY2RGB))
    upsampled_mask = cv.cvtColor(cv.resize(upsampled_mask,dsize=(512,384)),
            cv.COLOR_RGB2GRAY)

    return upsampled, upsampled_mask


def inpaint_dca(image, mask, i_type = 'ns'):
    """Inpaint the DCA region of the image
    Parameters
    ----------
    image : np.ndarray
        the image to inpaint
    mask : np.ndarray
        the corresponding mask for the image
    Returns
    -------
    np.ndarray
        the image with the DCA region inpainted
    """
    plt.figure(figsize=(5,5))
    plt.imshow(mask)
    plt.show()
    # Set inpaint type
    if i_type == 'ns':
        flags = cv.INPAINT_NS
    else:
        flags = cv.INPAINT_TELEA
    inpainted_image = cv.inpaint(image, mask.astype(np.uint8), inpaintRadius = 10, flags = flags)

    #Image.fromarray(inpainted_image).save(r'test.png')

    return inpainted_image


def remove_DCA(image, mask, removal_method = 'inpaint_ns'):
    """Remove DCA from a specified image. 
    
    Removal methods as follows:
        inpaint_ns = Navier Stokes method
        inpaint_telea = Telea method
    
    Duplicated as different return requirements metric generation
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
    # cropped_image, cropped_mask = reduce_intensity(image, mask)
    if image.shape[0] != 384 or image.shape[1] != 512:
        print('image shape not to scratch')
        # Only run this method if some of the image has been cropped
        # upsampled_image, upsampled_mask = run_super_resolution(image, mask)
        ##### remove next line for the above line when you are using a
        # computer that can handle it.
        inpainted_image = image
    else:
        # Otherwise pass the images through
        upsampled_image = image
        upsampled_mask = mask
        print("image shape unchanged")
        # remove when on a good compute
        inpainted_image = image
    
    # uncomment when on a good compute
    # if removal_method == 'inpaint_ns':
    #     inpainted_image = inpaint_dca(upsampled_image, upsampled_mask, 'ns')
    # elif removal_method == 'inpaint_telea':
    #     inpainted_image = inpaint_dca(upsampled_image, upsampled_mask, 'telea')
    # else:
    #     pass

    return inpainted_image
