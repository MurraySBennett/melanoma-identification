import math
import cv2
import numpy as np
from skimage.segmentation import chan_vese
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.pyplot as plt


class colour_balance:
    def __init__(self):
        pass

    def apply_mask(self, matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()

    def apply_threshold(self, matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = self.apply_mask(matrix, low_mask, low_value)

        high_mask = matrix > high_value
        matrix = self.apply_mask(matrix, high_mask, high_value)
        return matrix

    def simplest_cb(self, img, value):
        assert img.shape[2] == 3
        assert value > 0 and value < 100
        half_percent = value / 200.0
        channels = cv2.split(img)

        out_channels = []
        for channel in channels:
            assert len(channel.shape) == 2
            # find the low and high precentile values (based on the input percentile)
            height, width = channel.shape
            vec_size = width * height
            flat = channel.reshape(vec_size)

            assert len(flat.shape) == 1

            flat = np.sort(flat)

            n_cols = flat.shape[0]

            low_val = flat[math.floor(n_cols * half_percent)]
            high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

            # saturate below the low percentile and above the high percentile
            thresholded = self.apply_threshold(channel, low_val, high_val)
            # scale the channel
            normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
            out_channels.append(normalized)
        return cv2.merge(out_channels)

    def automatic_brightness_and_contrast(self, image, clip_hist_percent=1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return auto_result  # , alpha, beta)


def process_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hair removal code from sunnyshah2894
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (5, 5))
    # Perform blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # closing = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, kernel)
    # bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    # intensify the hair countours in preparation for the inpainting algorithm
    ret, thresh2 = cv2.threshold(blackhat, 5, 255, cv2.THRESH_BINARY)
    # inpaint the original image depending on the mask
    processed  = cv2.inpaint(img, thresh2, 6, cv2.INPAINT_TELEA)

    # return gray, blackhat, thresh2, dst
    return processed 

def crop_img(img, prop):
    # prop is the proportion of the image. The idea is that I will crop the edges before resizing it.
    # Note that this is an awful approach because there will be many cases where you crop useful parts of the img.
    h, w, channels = img.shape

    h_crop = int(np.round(h*prop/2))  # divide it by 2 because you're pinching from both ends.
    w_crop = int(np.round(w*prop/2))
    cropped_image = img[h_crop:h-h_crop, w_crop:w-w_crop]

    # # The below code tries to crop according to where the mask occurs. But I can't get the mask to pick out the
    # melanoma because of the dark edges. # https://www.sciencedirect.com/science/article/pii/S2215016120300832

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY) #
    # Find indices where we have mass mass_x, mass_y = np.where(thresh >= 255) # mass_x and mass_y are the list of x
    # indices and y indices of mass pixels cent_x = np.average(mass_x) cent_y = np.average(mass_y)
    #
    # # "major and minor axis of an ellipse that has the same second central moments as the inner area"
    # label_img = label(thresh)
    # regions = regionprops(label_img)
    # minr, minc, maxr, maxc = regions[0].bbox
    # cropped_image = img[minr:maxr, minc:maxc]
    # print(minr, maxr, minc, maxc, img.shape)
    # cropped_image = cv2.resize(cropped_image,size)
    return cropped_image


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # otsu thresholding
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Chan Vese segmentation
    cv = chan_vese(otsu, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                   max_num_iter=200, dt=0.5, init_level_set="disk",
                   extended_output=False)
    return otsu, cv


def super_pixel(img, n_segments):
    segments = slic(img, n_segments=n_segments, compactness=10, sigma=1)
    return segments


def impose_mask(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    return masked

def red_channel(img):
    B, G, R = cv2.split(img)

    BGR = np.sqrt(np.square(B) + np.square(G) + np.square(R))
    red_norm = np.divide(R, BGR)

    red_norm = np.where(red_norm == np.inf, 0, red_norm)
    # red_norm = cv2.merge([red_norm.astype(np.uint8), red_norm.astype(np.uint8), red_norm.astype(np.uint8)])
    # red_norm = cv2.cvtColor(red_norm, cv2.COLOR_BGR2GRAY)

    red_norm = cv2.merge([B.astype(np.uint8), G.astype(np.uint8), red_norm.astype(np.uint8)])

    return red_norm

def hsv_channel(img):
    return_img = img.copy()  # /.opy() so that you are making a new object
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    # shifted_H = np.mod(H, 180).astype(np.uint8)

    converted_img = S + V

    blur = cv2.GaussianBlur(converted_img, (3, 3), 0)
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # I don't think this adds anything
    # cv = chan_vese(otsu, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
    #                max_num_iter=200, dt=0.5, init_level_set="disk",
    #                extended_output=False)

    canny_h = cv2.Canny(H, 50, 120)
    canny_s = cv2.Canny(S, 100, 200)
    canny_v = cv2.Canny(V, 50, 100)
    canny_otsu = cv2.Canny(otsu, 100, 200)

    # Dilate img to 'close' spaces -- connect canny edges
    kernel = np.ones((3, 3), np.uint8)
    canny_otsu = cv2.dilate(canny_otsu, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(canny_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #  check contours
    w,h,d = img.shape
    new_contours = []
    for c in range(len(contours)):
        # remove small contours
        # if cv2.contourArea(contours[c]) < 40:
        #     continue
        # This check would remove 'outer' contours -- the 4th column represents the contour families.
        # A value of -1 indicates that it is the outermost contour of a group of contours. I think you WANT the outers.
        # But this is not the way to do it -- you may find the contour you're after exists within a larger one
        # (e.g., picture lens)
        # if hierarchy[0][c][3] < 0:
        #     continue

        # check for image edges - "remove" if the detected edge connects with the image edge
        if np.any(contours[c] == 0) or np.any(contours[c] == w-1):
            # cnt = contours[c]
            # cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
            continue
        new_contours.append(contours[c])

    # find largest contour and assume that it's the correct one
    cntsSorted = sorted(new_contours, key=lambda x: cv2.contourArea(x))
    largest = cntsSorted[-1]
    cv2.drawContours(return_img, [largest], 0, (255, 0, 0), 1)

    mask = np.zeros((w, h)).astype(np.uint8)
    cv2.fillPoly(mask, pts=[largest], color=(255, 255, 255))

    return mask, largest, return_img#, converted_img # largest contour

def get_shape_factor(contour):
    # https://pdf.sciencedirectassets.com/271303/1-s2.0-S0895611100X00948/1-s2.0-S0895611103000545/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGdm6zrlnd7Hg%2BH4W8hoik9TM3KjexKia0jQPetbBrQmAiEA%2Bdh08eEjgjDKdnfrm59LHYH9JUWPZ%2Bs8BDWB5PiJF4wqzAQIYBAFGgwwNTkwMDM1NDY4NjUiDDPrSqf5yEYJpHEZLiqpBB7JBs6FCxyYczj03OtJKqUhAQ0VPrDm9STHla1pAlSkJU93iMrI7hI%2FpeMANd0Vred2aWxKI9Ez0mZlbJDGFrZfm1A9I3QRU%2BLFmrwpJl%2BTxHVW3mGK09uqJBUWfX4IFpVQ0VG4czBnf497hYmR%2FMq1kugT5g8GeqgUTrE1jK7QCVnON5sCdk9Xc1s3DxqPmQd8hIFuJxp7Yda9R9eAg1R%2Fvaa0mxP1CVUVJy2aXo80zVS2m8MVVfRkKdOKqSbZdTdfVrgIpL2OD1zcYKbyzCFBuFyqhoAe%2Fp3vXHqUIN2I3ny3awe3ntzoCtYDhxqEFb%2FP5SYklGlUiYw%2B3AcWdPtwqfYHrw6MxLZxuiL7iogpzU7%2FSj%2FAvR%2F9p1k%2BISmsKB6o4GnpVKTsMDse%2BxTPdNLKygBI%2BPXYVUN3sTCBvY9XQXtgAbikMCMIVxJ7fOR%2BR4KB%2BNWC6KR0azM02iw4U3HX2p3bIvkyOKjly69MupoIzOJ48PgtZuUHwesUwxuRQhdLww5yYuq5x6SfCpIc2RTgdGMHvWLW%2FybVecVt7LorP9rLE1oMrL1IbJdKe71pdEIiYAD4KZdVTmwXz13W1b6QWClS%2Bvr5osVzxsahqs2rmj%2BiGOFvJ%2BphUeBiaCf%2FRXMtuptGgdqJZe86dzJ0ZXuq71URziX0eu4YPMsjSadC6OAcUnVrtnHokSZOVGoEESV%2BuR2NKLtreXU0XzOb6hebsncl%2FUOhFlQw6bC%2BnwY6qQFu2wNZL1zSDaSW5Sp4IaxEDswfyx71UwjYwVrw96nHV%2BYozq12YeO61yAi8Us0HHh0TD10y5OoaVVfsodTroEZVrtyWKx0lho9ge50hqEdjbOBU%2FQSb4cJym6LhfH26F0fGe7k2Du5PsUdEa44yVmSDXy0zLnM1k0trMoJBYUKhtOYgh%2BuT6CimR3WMRgbeJLAr0OtNUUO1nPTvnFJSYY2TikRc0lkC75h&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230217T161549Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3OXPVD2N%2F20230217%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=509164e150f9b7696987aca663bcaeed81bffdba0e0c906c9783bc8a3d168c81&hash=111506bd9b885b92221043a2e39709ef4733e44f45a8e56da27c93b3bb61c1df&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0895611103000545&tid=spdf-86a78e3a-2a79-4a69-a0f7-a4307bab079f&sid=5951b1b48f3ff34dcd2b2f624690359e5cedgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f11575d040800550604&rr=79afd2ee5ecd2cb4&cc=us
    # f = 4piA / P**2
    a = cv2.contourArea(contour)
    p = cv2.arcLength(contour, True)
    f = np.divide(4 * np.pi * a, p**2)
    return f




