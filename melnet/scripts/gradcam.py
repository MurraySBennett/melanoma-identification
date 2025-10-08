import tensorflow as tf
import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pprint import pprint as pp
from plotting import *
from glob import glob

strategy = tf.distribute.MirroredStrategy()
work = os.path.join(os.path.expanduser("~"), "win_home")
work_paths = dict(
    work=work,
    figures=os.path.join(work, "Documents", "melanoma-identification", "melnet", "figures"),
    images=os.path.join(work, "melanoma-identification", "images", "resized"),
    models=os.path.join(work,"Documents", "melanoma-identification", "melnet","models")
)

#feature_data = pd.read_csv(os.path.join(work_paths["data"], "btl-cv-data.csv"))
#fd_cols = ["id", "malignant"]
#feature_data = feature_data[fd_cols]

IMG_SIZE = 224
batch_size = 64


def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


sample_image = get_img_array(os.path.join(work_paths["images"], "ISIC_0000000.jpg"), size=(IMG_SIZE, IMG_SIZE))
last_conv_layer_name = "top_conv"

model_list = glob(os.path.join(work_paths["models"],'*'))
pp(model_list)

model = load_model(model_list[0])
model.layers[-1].activation = None
grad_model = keras.models.Model(
    model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
)
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)

heatmap = make_gradcam_heatmap(sample_image, model, last_conv_layer_name)

plt.matshow(heatmap)
plt.show()



