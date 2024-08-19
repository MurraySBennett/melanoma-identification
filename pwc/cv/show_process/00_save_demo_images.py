from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():
    """ save each step of the process"""
    home  = Path(__file__).resolve().parent.parent.parent
    paths = {
        "figures": home / "figures" / "cv",
        "images": home.parent / "images" / "resized" / "ISIC_0000003.JPG",
        "masks": home.parent / "images" / "masks" / "ISIC_0000003.png",
    }
    contour_colour = (12, 35, 64) # BGR tuple for "#0c2340"
    line_colour = (112, 90, 73) # BGR tuple for "#495a70" # dark gray used in poster
    line_colour = (212, 68, 13) # BGR tuple for orange
    linewidth = 5
    dash_length = 20
    gap_length = 15
    line_offset = 1

    img = cv.imread(paths["images"])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    original        = img.copy()
    counter = 0

    ########## Hair filter
    # filter_hair     = hair_rm(
    #     img.copy()
    # )
    filter_hair = img.copy()
    # reduce hair and skin texture artefacts 
    gray = cv.cvtColor(filter_hair, cv.COLOR_RGB2GRAY)
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
    hair_filter_images = {
        "Original" : original,
        "Grayscale": gray,
        "Identify Hair Contours": blackhat,
        "Gaussian Blur": bhg,
        "Intensify Hair Contours": thresh2,
        "Inpainting Contours": processed
    }
    for label, image in hair_filter_images.items():
        save_img(image, label, paths["figures"], counter)
        counter += 1


    ########## Colour clustering
    # cluster_colour  = colour_cluster(  
    #     hair_rm(
    #         img.copy()
    #     ),
    #     n_clusters = 6
    # )
    N_CLUSTERS = 6
    median = cv.medianBlur(processed, 5)
    Z = np.float32(median.reshape((-1, 3)))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = N_CLUSTERS
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((img.shape))

    clustering_images = {
        "Median Blur": median,
        "K-Means Clustering": kmeans_img
    }
    for label, image in clustering_images.items():
        save_img(image, label, paths["figures"], counter)
        counter += 1


    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    hsv = cv.cvtColor(kmeans_img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)
    h = clahe.apply(h)
    s = clahe.apply(s)
    v = clahe.apply(v)

    lab = cv.merge((h, s, v))
    enhanced = cv.cvtColor(lab, cv.COLOR_HSV2BGR)
    clahe_images = {
        "HSV Conversion": hsv,
        "CLAHE Enhanced": enhanced
    }
    for label, image in clahe_images.items():
        save_img(image, label, paths["figures"], counter)
        counter += 1

    ########## Segmentation and Mask Generation
    ### Canny OTSU thresholding
    hsv_img = cv.cvtColor(enhanced, cv.COLOR_BGR2HSV)
    _, S, V = cv.split(hsv_img)
    converted_img = S + V
    blur = cv.GaussianBlur(converted_img, (5,5), 0)
    _, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    canny_otsu_init = cv.Canny(otsu, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    canny_otsu = cv.dilate(canny_otsu_init, kernel, iterations=1)

    # get_contours()
    contours, _ = cv.findContours(canny_otsu, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        largest_contour = max(contours, key=cv.contourArea)
        contoured_img = cv.drawContours(
            image = original.copy(),
            contours = largest_contour,
            contourIdx = -1,
            color = contour_colour,
            thickness = 10
        )

    # contour_mask()
    h, w, _ = original.shape
    mask = np.zeros((h, w)).astype(np.uint8)
    cv.fillPoly(mask, pts=[largest_contour], color=(255, 255, 255))

    canny_otsu_images = {
        "HSV Conversion": hsv_img,
        "Combine SV Channels": converted_img,
        "Gaussian Blur": blur,
        "OTSU Threshold": otsu,
        "Canny Edge Detection": canny_otsu_init,
        "Edge Dilation": canny_otsu,
        "Largest Contour": contoured_img,
        "Mask": mask
    }
    for label, image in canny_otsu_images.items():
        save_img(image, label, paths["figures"], counter)
        counter += 1


    # shape analysis
    # trim edges -- some images have icons in corners
    h, w, _ = original.shape

    image_sym = original.copy()
    image_bor = original.copy()

    # Calculate the perimeter of the lesion contour
    perimeter = cv.arcLength(largest_contour, True)
    # Calculate the area of the lesion contour
    area = cv.contourArea(largest_contour)
    # Calculate the radius of the circle with the same perimeter as the contour
    radius = int(perimeter / (2 * np.pi))

    # get the central point of the contour
    M = cv.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # Find the maximum and minimum y positions of the contour
    minY = min(largest_contour[:, 0, 1])
    maxY = max(largest_contour[:, 0, 1])
    minX = min(largest_contour[:, 0, 0])
    maxX = max(largest_contour[:, 0, 0])

    newCenterY = (minY + maxY) // 2
    newCenterX = (minX + maxX) // 2

    # Draw the lesion contour
    cv.drawContours(image_sym, [largest_contour], 0, contour_colour, linewidth)
    cv.drawContours(image_bor, [largest_contour], 0, contour_colour, linewidth)

    # Draw the circle with the same perimeter from the (self-identified) center of the contour
    cv.circle(image_bor, (newCenterX, newCenterY), radius-line_offset, contour_colour, linewidth)
    cv.circle(image_bor, (newCenterX, newCenterY), radius+line_offset, contour_colour, linewidth)
    cv.circle(image_bor, (newCenterX, newCenterY), radius, line_colour, linewidth)

    # Crop image to center the lesion and circle
    # Define the buffer size as a percentage of the circle radius
    buffer_percentage = 0.2  # 10% buffer
    # Calculate the buffer size in pixels
    buffer_size = int(radius * buffer_percentage)

    # Calculate the new width and height of the cropped region with the buffer
    newWidth = newHeight = 2 * (radius + buffer_size)

    # Calculate the top-left corner of the cropped region
    # cropX = cX - (radius + buffer_size)
    # cropY = newCenterY - (radius + buffer_size)

    # Crop the image to the specified region
    # cropped_sym = image_sym[cropY:cropY+newHeight, cropX:cropX+newWidth]
    # cropped_bor = image_bor[cropY:cropY+newHeight, cropX:cropX+newWidth]

    # compactness = np.divide(4 * np.pi * area, perimeter**2) 
    minY = min(largest_contour[:, 0, 1])# - cropY
    maxY = max(largest_contour[:, 0, 1])# - cropY
    minX = min(largest_contour[:, 0, 0])# - cropX
    maxX = max(largest_contour[:, 0, 0])# - cropX
    newCenterY = (minY + maxY) // 2
    newCenterX = (minX + maxX) // 2


    ####### asymmetry
    # Draw the dashed vertical line
    current_length = 0
    while current_length < (maxY - minY):
        if current_length % (dash_length + gap_length) < dash_length:
            cv.line(image_sym, (newCenterX-line_offset, minY + current_length - line_offset), (newCenterX+line_offset, minY + current_length + line_offset), contour_colour, linewidth+(line_offset*2))
        current_length += 1

    current_length = 0
    while current_length < (maxY - minY):
        if current_length % (dash_length + gap_length) < dash_length:
            cv.line(image_sym, (newCenterX, minY + current_length), (newCenterX, minY + current_length), line_colour, linewidth)
        current_length += 1


    # colour analysis
    colour_img = cv.imread(paths["images"])
    # original uses Lab colour space, but you do RGB here to display example.
    colour_img = cv.cvtColor(colour_img, cv.COLOR_BGR2RGB)
    # mask = cv.imread(paths["masks"], -1)
    segmented_lesion = cv.bitwise_and(colour_img, colour_img, mask=mask.astype(np.uint8))

    colour_analysis = {
        "Shape Asymmetry": image_sym,
        "Border Irregularity": image_bor,
        "Colour Variance": segmented_lesion,
    }
    for label, image in colour_analysis.items():
        save_img(image, label, paths["figures"], counter)
        counter += 1




def save_img(image, label, path, counter):
    """ save images """
    print(f"Saving {label}...")
    grayscale_labels = [
        "Grayscale",
        "Identify Hair Contours",
        "Gaussian Blur",
        "Intensify Hair Contours",
        "Combine SV Channels",
        "OTSU Threshold",
        "Canny Edge Detection",
        "Edge Dilation",
        "Largest Contour",
        "Mask",
    ]

    plt.figure(figsize=(4,4))
    if label in grayscale_labels:
        plt.imshow(image, cmap="Greys")
    else:
        plt.imshow(image)
    plt.axis("off")
    plt.title(label, fontsize=24)
    plt.tight_layout()
    counter = "0" + str(counter) if counter < 10 else str(counter)
    plt.savefig(
        fname = path / f"{counter}_{label.replace(" ", "_")}.pdf",
        format = "pdf", bbox_inches = "tight"
    )
    plt.savefig(
        fname = path / f"{counter}_{label.replace(" ", "_")}.png",
        format = "png", dpi=600, bbox_inches = "tight"
    )
    plt.close()


def process_img(img, n_clusters=6, segment_type='otsu', save_img_label=None):
    """ combine all functions for full processing method """
    logger = set_logger()
    try:
        original = img.copy()
        # dca = get_dca(img)
        # img = paint_dca(dca[0], dca[1], method='inpaint_ns')
        img = hair_rm(img)
        img = colour_cluster(img, n_clusters)
        img = my_clahe(img)
        if segment_type == 'otsu':
            mask = otsu_segment(original, img)
        elif segment_type == 'grabcut':
            mask = grabcut_segment(original, img)
        if save_img_label is not None:
            save_img_label(mask, save_img_label)
        return mask
    except Exception as e:
        logger.error('An error occurred: %s', e)


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


def colour_cluster(img, n_clusters):
    """ cluster image colours for improved border detection """
    median = cv.medianBlur(img, 5)
    Z = np.float32(median.reshape((-1, 3)))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n_clusters
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((img.shape))
    return kmeans_img


def my_clahe(img):
    """ contrast limited adaptive histogram equalisation.
    returns enhanced image """
    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = clahe.apply(h)
    s = clahe.apply(s)
    v = clahe.apply(v)

    lab = cv.merge((h, s, v))
    enhanced = cv.cvtColor(lab, cv.COLOR_HSV2BGR)
    return enhanced


def otsu_thresh(img):
    """ threshold image using otsu method and return canny edge detection """
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_img)
    converted_img = S + V
    blur = cv.GaussianBlur(converted_img, (5,5), 0)
    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    canny_otsu = cv.Canny(otsu, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    canny_otsu = cv.dilate(canny_otsu, kernel, iterations=1)
    return canny_otsu


def get_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    new_contours=[]
    if len(contours) != 0:
        largest_contour = max(contours, key=cv.contourArea)

    return largest_contou 


def contour_mask(img, contour):
    h, w, d = img.shape
    mask = np.zeros((h, w)).astype(np.uint8)
    cv.fillPoly(mask, pts=[contour], color=(255, 255, 255))
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
    _, segmented_mask = cv.threshold(img_mask, 0, 255, cv.THRESH_BINARY)
    return segmented_mask


def otsu_segment(img, enhanced):
    """ otsu thresholding """
    canny_otsu = otsu_thresh(enhanced)
    largest_contour = get_contours(canny_otsu)
    otsu_mask = contour_mask(img, largest_contour)
    return otsu_mask


main()