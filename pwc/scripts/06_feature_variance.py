import matplotlib
matplotlib.use('Agg') # Ensure headless compatibility
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Constants for consistency
TILE_SIZE = (256, 256)
BORDER_SIZE = 3

from ..config import (FILES, PATHS)

data_path   = FILES['btl_cv']
img_path    = PATHS['images']
figure_path = PATHS['figures']
data = pd.read_csv(data_path)

features = ['pi_sym', 'pi_bor', 'pi_col']
data = data[['id', 'malignant'] + features].dropna()

feature_labels = ["Asymmetry", "Border\nIrregularity", "Colour\nVariance"]
exemplars = {}

# font = FontProperties(fname="Garamond BoldCondensed.ttf")
font = FontProperties()
FONT_COLOUR = "black" #"#0c2340" # the hex code for contour_colour
FONT_SIZE = 24
LINE_COLOUR = "black"#"#D4440D"
AXIS_LABEL_FONT_SIZE = 18

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42

def load_img(f, im_path):
    """Loads, resizes, and adds a white border."""
    # Robust file finding
    target = im_path / f"{f}.JPG"
    if not target.exists():
        target = im_path / f"{f}.jpg"
    
    img = cv2.imread(str(target))
    if img is None:
        # Return a grey placeholder if image is missing
        return np.full((*TILE_SIZE, 3), 128, dtype=np.uint8)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TILE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Add border using slicing (more efficient than manual loops)
    img[:BORDER_SIZE, :, :] = 255
    img[-BORDER_SIZE:, :, :] = 255
    img[:, :BORDER_SIZE, :] = 255
    img[:, -BORDER_SIZE:, :] = 255
    return img

def get_pctile_deterministic(df, feature, p, malignant=0):
    """Returns the image closest to the p-th percentile with deterministic tie-breaking."""
    # Subset by malignancy
    sub = df[df["malignant"] == malignant].copy()
    
    # Sort by feature AND ID to ensure tie-breaking is identical every time
    sub = sub.sort_values(by=[feature, 'id']).reset_index(drop=True)
    
    if sub.empty:
        return np.zeros((*TILE_SIZE, 3), dtype=np.uint8)

    # Calculate index based on percentile
    idx = int(len(sub) * (p / 100))
    idx = min(idx, len(sub) - 1)
    
    img_id = sub.iloc[idx]['id']
    return load_img(img_id, img_path)

def main():
    # Load data and ensure it is sorted once globally
    data = pd.read_csv(data_path).dropna(subset=features + ['malignant'])
    
    n_images = 5
    # Standardize percentiles (0 to 100)
    pctiles = np.linspace(5, 95, n_images) 
    
    fig, axs = plt.subplots(3, 1, figsize=(16, 8))

    for i, feature in enumerate(features):
        row_images = []
        for p in pctiles:
            # Logic: Use non-malignant for lower pctiles, malignant for higher if desired
            # or keep it consistent across the row.
            m_status = 0 if p < 67 else 1
            img = get_pctile_deterministic(data, feature, p, malignant=m_status)
            row_images.append(img)
            
        final_image = np.hstack(row_images)
        axs[i].imshow(final_image)
        
        # Formatting
        axs[i].set_ylabel(feature_labels[i], fontsize=FONT_SIZE)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        for spine in axs[i].spines.values():
            spine.set_visible(False)

    axs[0].set_title(r'Less $\longleftarrow$ Feature Gradient $\longrightarrow$ More', fontsize=FONT_SIZE)
    
    plt.tight_layout()
    plt.savefig(figure_path / "feature_variance.pdf", bbox_inches='tight', dpi=300)
    print("Figure saved successfully.")

if __name__ == "__main__":
    main()
    
