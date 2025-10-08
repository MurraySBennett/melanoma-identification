# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:42:09 2025

@author: bennett.1755
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

home_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
paths = {
    'home': home_dir,
    'images': home_dir / 'images' / 'resized',
    'scripts': home_dir / 'grt' / 'scripts', 
    'figures': home_dir / 'grt' / 'figures',
    'data': home_dir / 'pwc' / 'data' / 'estimates' / 'btl_data_revised.csv'
    }

df = pd.read_csv(paths['data'])

# 2. Identify high and low values (using percentiles for demonstration)
qtile = 0.85
pi_sym_low = df['pi_sym'].quantile(1-qtile)
pi_sym_high = df['pi_sym'].quantile(qtile)
pi_col_low = df['pi_col'].quantile(1-qtile)
pi_col_high = df['pi_col'].quantile(qtile)

# 3. Filter data
ll_df = df[(df['pi_sym'] <= pi_sym_low) & (df['pi_col'] <= pi_col_low) & (df['malignant'] == 0)]
lh_df = df[(df['pi_sym'] <= pi_sym_low) & (df['pi_col'] >= pi_col_high) & (df['malignant'] == 0)]
hl_df = df[(df['pi_sym'] >= pi_sym_high) & (df['pi_col'] <= pi_col_low) & (df['malignant'] == 0)]
hh_df = df[(df['pi_sym'] >= pi_sym_high) & (df['pi_col'] >= pi_col_high) & (df['malignant'] == 0)]

# 4. Create the figure
fig, ax = plt.subplots(figsize=(8, 8))

# Cross in the center
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# # Quadrant labels
# ax.text(-0.5, -0.9, "Low-Low", ha='center', va='top', fontsize=12)
# ax.text(-0.5, 0.9, "Low-High", ha='center', va='top', fontsize=12)
# ax.text(0.5, -0.9, "High-Low", ha='center', va='top', fontsize=12)
# ax.text(0.5, 0.9, "High-High", ha='center', va='top', fontsize=12)


# Row and column headings
ax.text(-1.01, -0.5, "Low", ha='right', va='center', rotation=90, fontsize=16)
ax.text(-1.01, 0.5, "High", ha='right', va='center', rotation=90, fontsize=16)
ax.text(-0.5, 1.01, "Low", ha='center', va='bottom', fontsize=16)
ax.text(0.5, 1.01, "High", ha='center', va='bottom', fontsize=16)

# Dimension headings
ax.text(-1.1, 0., "Colour Variance", ha='right', va='center', rotation=90, fontsize=20)
ax.text(-0.0, 1.1, "Asymmetry", ha='center', va='bottom', fontsize=20)


# Place images
idx = 2 # really, just use 0
if not ll_df.empty:
    ll_image_id = ll_df.iloc[idx]['id']
    ll_image_path = paths['images'] / f"{ll_image_id}.JPG"
    try:
        ll_image = plt.imread(ll_image_path)
        ax.imshow(ll_image, extent=[-1, 0, -1, 0])
    except FileNotFoundError:
        print(f"Warning: Image not found for {ll_image_id}")

if not lh_df.empty:
    lh_image_id = lh_df.iloc[idx]['id']
    lh_image_path = paths['images'] / f"{lh_image_id}.JPG"
    try:
        lh_image = plt.imread(lh_image_path)
        ax.imshow(lh_image, extent=[-1, 0, 0, 1])
    except FileNotFoundError:
        print(f"Warning: Image not found for {lh_image_id}")

if not hl_df.empty:
    hl_image_id = hl_df.iloc[idx]['id']
    hl_image_path = paths['images'] / f"{hl_image_id}.JPG"
    try:
        hl_image = plt.imread(hl_image_path)
        ax.imshow(hl_image, extent=[0, 1, -1, 0])
    except FileNotFoundError:
        print(f"Warning: Image not found for {hl_image_id}")

if not hh_df.empty:
    hh_image_id = hh_df.iloc[idx]['id']
    hh_image_path = paths['images'] / f"{hh_image_id}.JPG"
    try:
        hh_image = plt.imread(hh_image_path)
        ax.imshow(hh_image, extent=[0, 1, 0, 1])
    except FileNotFoundError:
        print(f"Warning: Image not found for {hh_image_id}")

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')


# 5. Save or display the figure
plt.savefig(paths['figures'] / 'demo_levels.pdf')
plt.show() #use this instead of savefig if you want to display the image.



