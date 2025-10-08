import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def main():
    
    n_images = None # None = all
    
    base_dir = Path(os.getcwd()).parent.parent
    model_path = base_dir / 'melnet' / 'models' / 'melnet'
    weights_path = base_dir / 'melnet' / 'models'
    images_path = base_dir / 'images' / 'resized'
    data_path = base_dir / 'melnet' / 'data'
    mal_path = base_dir / 'pwc' / 'data' / 'estimates' / 'btl_cv_data_revised.csv'
    
    mal_data = pd.read_csv(mal_path)
    mal_data["id"] = [i.split(".")[0] for i in mal_data["id"]]
    
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg'))]  
    image_ids = [f.split(".")[0] for f in image_files]
    valid = mal_data["id"].tolist()
    if n_images is not None:
        valid = valid[:n_images]
    
    image_files = [image_files[i] for i, img_id in enumerate(image_ids) if img_id in valid]    
    loaded_images, loaded_ids = load_images_threaded(image_files, images_path)
    
    model = tf.keras.models.load_model(model_path)
   
    all_predictions = pd.DataFrame({"id": loaded_ids})
    all_predictions['id'] = [img.split('.')[0] for img in all_predictions['id']]
    all_predictions = pd.merge(all_predictions, mal_data[['id', 'malignant']], left_on='id', right_on='id', how='left')
    
    best_model_prob_scores = []
    for img in tqdm(loaded_images, desc="Predicting images with best model"):
        best_model_prob_scores.append(predict_image(img, model))
    all_predictions['probability_best'] = np.round(best_model_prob_scores, 4)
    
    # Extract and save activations for the best model
    batch_size = 32
    all_activations = []
    with tqdm(total=len(loaded_images) // batch_size + (1 if len(loaded_images) % batch_size else 0), desc="Extracting activations") as pbar:
        for i in range(0, len(loaded_images), batch_size):
            batch_images = loaded_images[i:i + batch_size]
            batch_images_np = np.concatenate(batch_images, axis=0)
            batch_activations = extract_activations(model, batch_images_np)
            all_activations.append(batch_activations)
            pbar.update(1)
    activations = np.concatenate(all_activations, axis=0)
    activations_df = pd.DataFrame(activations)
    activations_df["id"] = [i.split(".")[0] for i in loaded_ids]
    
    cols = activations_df.columns.tolist()
    cols = ['id'] + [c for c in cols if c != 'id']
    activations_df = activations_df[cols]
    
    activations_df.to_csv(data_path / "activations" / "best_model_avg_pool_activations.csv", index=False)
    print(f"Activations for best model saved to: {data_path / 'activations' / 'best_model_avg_pool_activations.csv'}")

    # rm last - it is presently the best model, anyway
    weights = list(weights_path.glob("weights*.h5"))[:-1]
    for w in weights:
        fname_parts = w.name.split('_')
        if len(fname_parts) > 1:
            save_label = '_'.join(fname_parts[1:]).split('.h5')[0]
        else:
            save_label = fname_parts[-1].split(".h5")[0]
        print(save_label)
    
        model.load_weights(w)
        prob_scores = []
        for img in tqdm(loaded_images, desc=f"Predicting images with {save_label} weights"):
            prob_scores.append(predict_image(img, model))
        all_predictions[f'probability_{save_label}'] = np.round(prob_scores, 4)
    
    all_predictions.to_csv(data_path / 'predictions' / 'all_model_predictions.csv', index=False)
    print(f"All predictions saved to: {data_path / 'predictions' / 'all_model_predictions.csv'}")
        
        
def load_image(image_path):
    IMG_SIZE = 224
    try:
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def load_images_threaded(image_files, images_path):
    image_paths = [os.path.join(images_path, f) for f in image_files]
    loaded_images = [None] * len(image_files)
    image_ids = {}
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, path): i for i, path in enumerate(image_paths)}
        for future in tqdm(as_completed(futures), desc="Loading images"):
            index = futures[future]
            loaded_images[index] = future.result()
            image_ids[index] = image_files[index]
    
    final_images = [img for img in loaded_images if img is not None]
    final_ids = [image_ids[i] for i, img in enumerate(loaded_images) if img is not None]
    
    return final_images, final_ids


def predict_image(image, model):
    try:
        predictions = model.predict(image, verbose=0)
        return predictions[0][0]
    except Exception as e:
        print(f"Error predicting on image: {e}")
        return None
    

def extract_activations(model, images, layer_name="avg_pool"):
    intermediate_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = intermediate_layer.predict(images, verbose=0)
    return np.round(activations, 3)


    

if __name__ == "__main__":
    main()
