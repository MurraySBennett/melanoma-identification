from pathlib import Path
import matplotlib
matplotlib.use("Pdf")  # Headless-safe backend for HPC
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns

from .cv_transforms import abc_aligned, cv_btl_scale
from ..config import (FILES, PATHS)

# Global Publication Styling
plt.rcParams.update({
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
    'font.family': 'sans-serif', 
    'font.sans-serif': ['Arial'],
    'text.antialiased': True
})

# Formatting Constants
FONT_COLOUR = "black"
FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TEXT_FONT_SIZE = 16
COLOUR_B = '#0c2340'  # Benign
COLOUR_M = '#D4440D'  # Malignant

def get_rho(x, y):
    """Calculates Spearman Rho and handles NaNs deterministically."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(mask):
        return {'r': np.nan, 'p': np.nan}
    rho, p = spearmanr(x[mask], y[mask])
    return {'r': rho, 'p': p}

def get_ls_data(x, y):
    """Calculates Linear Regression line based on valid pairwise points."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_val = x[mask].reshape(-1, 1)
    y_val = y[mask].reshape(-1, 1)
    
    if x_val.size == 0:
        return None, None
        
    model = LinearRegression().fit(x_val, y_val)
    # Create x-values for the line spanning the actual data range
    x_range = np.linspace(x_val.min(), x_val.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    return x_range, y_pred

def main():
    # 1. Data Preparation
    data_raw = pd.read_csv(FILES["btl_cv"])
    data = abc_aligned(data_raw)
    data = cv_btl_scale(data, replace=True)
    
    features_btl = ["pi_sym", "pi_bor", "pi_col"]
    features_cv = ["sym", "bor", "col"]
    f_labels = ['Asymmetry', 'Border Irregularity', 'Colour Variance']
    
    # 2. Correlation Matrix Calculation
    # We use Spearman for everything to remain consistent across features and malignancy
    cor_cols = features_btl + features_cv + ["malignant"]
    cor_features = data[cor_cols].corr(method='spearman')

    # 3. Figure 1: Triple Scatter Plot
    PLT_SIZE = 5
    fig, axes = plt.subplots(1, 3, figsize=(PLT_SIZE*3, PLT_SIZE))
    
    for i, label in enumerate(f_labels):
        ax = axes[i]
        x_data = data[features_cv[i]].values
        y_data = data[features_btl[i]].values
        
        # Plot Scatter by Diagnosis
        for diag_val, diag_label, col, mark in zip([0, 1], ['Benign', 'Malignant'], [COLOUR_B, COLOUR_M], ['o', '^']):
            mask = data['malignant'] == diag_val
            # Subsampling for clarity if dataset is large (STEP_SIZE=10)
            ax.scatter(x_data[mask][::10], y_data[mask][::10], 
                       s=30, marker=mark, color=col, label=diag_label if i == 0 else "", alpha=0.7)

        # Regression Line
        x_line, y_line = get_ls_data(x_data, y_data)
        if x_line is not None:
            ax.plot(x_line, y_line, color='black', linewidth=1.5, linestyle='--')

        # Stats Annotation
        stats = get_rho(x_data, y_data)
        p_str = "p < .001" if stats['p'] < 0.001 else f"p = {stats['p']:.3f}"
        stats_txt = f"r = {stats['r']:.3f}\n{p_str}"
        ax.text(0.95, 0.05, stats_txt, transform=ax.transAxes, fontsize=TEXT_FONT_SIZE,
                verticalalignment='bottom', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_title(label, fontsize=FONT_SIZE)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.spines[["top", "right"]].set_visible(False)
        
        if i == 1:
            ax.set_xlabel('Computer Vision Estimates', fontsize=AXIS_LABEL_FONT_SIZE)
        if i == 0:
            ax.set_ylabel('Perceptual Strength (BTL)', fontsize=AXIS_LABEL_FONT_SIZE)
            ax.legend(loc='upper left', fontsize=TEXT_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(PATHS['figures'] / "btl_cv_cor_scatters.pdf", bbox_inches='tight')

    # 4. Figure 2: Correlation Heatmap
    sns.set_theme(style='ticks')
    plt.figure(figsize=(7, 6))
    mask = np.triu(np.ones_like(cor_features, dtype=bool))
    
    # Mapping shortened labels for publication clarity
    short_labels = ['A', 'B', 'C', 'CV_A', 'CV_B', 'CV_C', 'Malig.']
    
    sns.heatmap(cor_features,
                annot=True, fmt=".2f", square=True, linewidths=.5,
                mask=mask, cmap='coolwarm', vmin=0, vmax=1, center=0.5,
                xticklabels=short_labels, yticklabels=short_labels,
                cbar_kws={"shrink": .8})
    
    plt.title("Spearman Correlation Matrix", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(PATHS["figures"] / "btl_cv_cor_matrix.pdf", bbox_inches="tight")
    
    print(f"Figures saved to {PATHS['figures']}")

if __name__ == "__main__":
    main()
