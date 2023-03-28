# https://3b1b.github.io/manim/

# https://docs.manim.community/en/stable/tutorials/output_and_config.html
# https://docs.manim.community/en/stable/guides/configuration.html
#  to render, enter the following into the cmd line: manim -pql file_name.py
# quality flags:
# ql, qm, -qh, and -qk
# note: good to check your work with the -ql flag, then set to a higher level when ready.

# flags:
# -p: play video once it renders
# -a: render (a)ll scenes in the file (you can include some scenes, but select only the ones you want rendered by
# calling the name of the scene after the filename. For example: manim -pql file_name.py squaretocircle
# -p: opens the file location of the rendered video
# -i: save as gif, instead of mp4

from manim import *
import os
import numpy as np
from PIL import Image
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer, Convolutional2DLayer, ImageLayer, MaxPooling2DLayer

anim_run_time = 15 # seconds

img_path = os.path.join(os.getcwd(), "images", "ISIC-2020", "ISIC_0000000.JPG")
image = Image.open(img_path)  # You will need to download an image of a digit.
numpy_image = np.asarray(image)

# This changes the resolution of our rendered videos
config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0


# Here we define our basic scene
class CNN(ThreeDScene):
    # The code for generating our scene goes here
    def construct(self):
        # Make the neural network
        # num_feature_maps, feature_map_size, filter_size
        # num_feature_maps = number of slices,
        # feature_map_size = n rows and columns of the feature map
        # filter_size = Number of feature_map rows/cols evaluated on each iteration
        # If filer_size == 3, the feature map of the next level should be -2.

        nn = NeuralNetwork([
            ImageLayer(numpy_image, height=1.5),
            # Feature map sizes and filter dimensions of adjacent layers match up
            Convolutional2DLayer(num_feature_maps=1, feature_map_size=6, filter_size=3, filter_spacing=0.32),  # Note the default stride is 1.
            Convolutional2DLayer(num_feature_maps=3, feature_map_size=4, filter_size=3, filter_spacing=0.25),
            MaxPooling2DLayer(kernel_size=2),
            Convolutional2DLayer(3, 3, 3, filter_spacing=0.25),

            FeedForwardLayer(3),
            FeedForwardLayer(6),
            FeedForwardLayer(2),
        ],
            layer_spacing=0.25,
        )
        # Center the neural network
        nn.move_to(LEFT)
        self.add(nn)
        # Make a forward pass animation
        forward_pass = nn.make_forward_pass_animation()

        # Play animation
        self.play(forward_pass, run_time=anim_run_time)
