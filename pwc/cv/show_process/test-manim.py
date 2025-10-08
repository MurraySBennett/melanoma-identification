from manim import *
import cv2 as cv
import numpy as np
import os
import glob
# import matplotlib.pyplot as plt
from image_processes import *

# class GradientImageFromArray(Scene):
#     def construct(self):
#         n = 256
#         imageArray = np.uint8(
#             [[i * 256 / n for i in range(0, n)] for _ in range(0, n)]
#         )
#         image = ImageMobject(imageArray).scale(2)
#         image.background_rectangle = SurroundingRectangle(image, GREEN)
#         self.add(image, image.background_rectangle)
#         self.wait(3)


# f, ax = plt.subplots(1,1, figsize=(5,5))
# ax.imshow(original)
# ax.axis('off')
# f.tight_layout()
# plt.show()

class ImageProcessingPipeline(Scene):
    def construct(self):
        home_path = os.path.expanduser('~')
        paths = dict(home=home_path,
                images=os.path.join(home_path, 'win_home', 'Documents', 'example-images'))
        image_path = glob.glob(os.path.join(paths['images'], '*0.JPG'))

        original = cv.imread(image_path[0])
        og_copy = original.copy()
        sans_hair= hair_rm(original)
        colour_transform = my_clahe(sans_hair)

        contoured_clrTrns = colour_transform.copy()
        canny_edge = otsu_thresh(contoured_clrTrns)
        contour = get_contours(canny_edge)
        cv.drawContours(contoured_clrTrns, contour, -1, (40, 240, 40), thickness=10)

        segment = otsu_segment(og_copy, colour_transform)
        
        original = ImageMobject(original).scale(2)
        sans_hair = ImageMobject(sans_hair).scale(2)
        colour_transform = ImageMobject(colour_transform).scale(2)
        contoured_clrTrns = ImageMobject(contoured_clrTrns).scale(2)
        segment = ImageMobject(segment).scale(2)

        self.add(original)
        self.play(Transform(original, sans_hair), run_time=1)
        self.play(Transform(sans_hair, colour_transform), run_time=1)
        self.play(Transform(colour_transform,contoured_clrTrns), run_time=1)
        self.play(Transform(contoured_clrTrns, segment), run_time=1)
        # self.wait(2)


