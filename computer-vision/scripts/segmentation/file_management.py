import cv2 as cv
import os

def save_img(img, path, label):
    cv.imwrite(os.path.join(path, label) + '.jpg', img)
