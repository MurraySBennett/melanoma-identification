import pandas as pd
import numpy as np
import cv2 as cv
import os, os.path
import matplotlib.pyplot as plt


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
