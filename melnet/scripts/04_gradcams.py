import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm

IMG_SIZE = 224

def main():
    base_dir = Path(os.getcwd()).parent.parent
    model_path = base_dir / 'melnet' / 'models' / 'melnet'
    weights_path = base_dir / 'melnet' / 'models'
    figures_path = base_dir / 'melnet' / 'figures' / 'gradcams'
    images_dir = base_dir / 'images' / 'resized'

    img_list = {
        'ambiguous': ['ISIC_0011030.JPG', 'ISIC_0033584.JPG', 'ISIC_0068223.JPG', 'ISIC_0024277.JPG'],
        'confident': ['ISIC_0000030.JPG', 'ISIC_0000031.JPG', 'ISIC_0000186.JPG', 'ISIC_0002213.JPG',]
        }

    model = load_model(model_path)
    model.layers[-1].activation = None

    process_images(model, img_list['ambiguous'], images_dir, figures_path / 'ambiguous', "top_conv", save_label="best")
    process_images(model, img_list['confident'], images_dir, figures_path / 'confident', "top_conv", save_label="best")

    weights = list(weights_path.glob("weights*.h5"))[:-1]
    for w in weights:
        model.load_weights(w)
        fname_parts = w.name.split('_')
        if len(fname_parts) > 1:
            save_label = '_'.join(fname_parts[1:]).split('.h5')[0]
        else:
            save_label = fname_parts[-1].split(".h5")[0]
        process_images(model, img_list['ambiguous'], images_dir, figures_path / 'ambiguous', "top_conv", save_label=save_label)
        process_images(model, img_list['confident'], images_dir, figures_path / 'confident', "top_conv", save_label=save_label)
        

def get_img_array(img_path, size):
    try:
        img = load_img(img_path, target_size=size)
        array = img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(original_img_path, heatmap, save_path, alpha=0.5):
    try:
        img = load_img(original_img_path)
        img = img_to_array(img)

        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("viridis")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        superimposed_img.save(save_path)
    except Exception as e:
        print(f"Error saving Grad-CAM for {original_img_path}: {e}")


def process_images(model, img_list, images_dir, figures_dir, layer_name="top_conv", save_label=""):
    for img_name in tqdm(img_list, desc="Generating Grad-CAMs"):
        img_path = os.path.join(images_dir, img_name)
        img_array = get_img_array(img_path, size=(IMG_SIZE, IMG_SIZE))

        if img_array is not None:
            heatmap = make_gradcam_heatmap(img_array, model, layer_name)
            base_filename, extension = os.path.splitext(img_name)
            save_path = os.path.join(figures_dir, f"{base_filename}_gradcam_{save_label}{extension}")
            save_gradcam(img_path, heatmap, save_path)


if __name__ == '__main__':
    main()