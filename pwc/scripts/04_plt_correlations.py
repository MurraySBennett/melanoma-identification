from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from cv_transforms import abc_aligned, cv_btl_scale

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

font = FontProperties()
FONT_COLOUR = "black" # the hex code for contour_colour
FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TEXT_FONT_SIZE = 16
plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42

home = Path(__file__).resolve().parent.parent
paths = dict(
    figures = home / "figures",
    data = home / "data" / "estimates" / "btl-cv-data.csv"
)
data = pd.read_csv(paths["data"])
data = abc_aligned(data)
data = cv_btl_scale(data, replace=True)
data = data[["id", "malignant", "pi_sym", "pi_bor", "pi_col", "sym", "bor", "col"]]

features_btl = ["pi_sym", "pi_bor", "pi_col"]
features_cv = ["sym", "bor", "col"]

hm_data = data[features_btl + features_cv]
cor_features = hm_data.corr(method='spearman')
cor_mal = data[features_btl + features_cv + ["malignant"]].corr()
# add Pearson column then Pearson row for malignancy
cor_features['malignant'] = cor_mal['malignant']
cor_features.loc['malignant'] = cor_mal['malignant']

data["Diagnosis"] = data["malignant"].replace({0:"Benign", 1:"Malignant"})
malignant = ["Diagnosis"]


def get_rho(x, y):
    rho, p = spearmanr(x, y, nan_policy='omit')
    return {'r': rho, 'p': p, 'rp': str(np.round(rho,3))+'\n'+str(np.round(p,3))}


def get_ls(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    imputer = SimpleImputer(strategy='mean')
    y = imputer.fit_transform(y)

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    pred = slope * x + intercept
    return {"slope": slope, "intercept": intercept, "lsline": pred}


COLOUR_B = '#0c2340'
COLOUR_M = '#D4440D'
SHOW_FIG=True
if SHOW_FIG:
    colours = {'Benign': COLOUR_B, 'Malignant': COLOUR_M}
    markers = ['o','^']
    # plt.figure(figsize=(8,6))
    # sns.heatmap(
        # cor_mat, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, mask=mask
    # )
    # plt.title("Correlation Heatmap")
    # plt.savefig(
        # path.join(paths["figures"], 'btl-cv-cor-heatmap.pdf'),
        # format='pdf', dpi=600, bbox_inches='tight'
    # )

    # STEP_SIZE = 20
    # sns_plt = sns.pairplot(
        # data[features_btl + features_cv + malignant][::STEP_SIZE],
        # hue="Diagnosis", palette=colours, markers=markers,
        # corner=True, height=1, diag_kind='auto'
    # )
    # sns_plt.savefig(
        # path.join(paths["figures"], 'btl_cv_cor_full.pdf'),
        # format='pdf', dpi=600, bbox_inches='tight'
    # )

    # sns_plt = sns.pairplot(
        # data[features_cv + malignant][::STEP_SIZE],
        # hue="Diagnosis", palette=colours, markers=markers,
        # corner=True, height=1.5
    # )
    # sns_plt.savefig(
        # path.join(paths["figures"], 'btl_cor.pdf'),
        # format='pdf', dpi=600, bbox_inches='tight'
    # )

    # sns_plt = sns.pairplot(
        # data[features_btl + malignant][::STEP_SIZE],
        # hue="Diagnosis", palette=colours, markers=markers,
        # corner=True, height=1.5
    # )
    # sns_plt.savefig(
        # path.join(paths["figures"], 'cv_cor.pdf'),
        # format='pdf', dpi=600, bbox_inches='tight'
    # )

    features    = [features_btl + features_cv]
    f_labels    = ['Asymmetry', 'Border Irregularity', 'Colour Variance']
    malignancy  = ['Benign', 'Malignant']
    colours     = {0:COLOUR_B, 1:COLOUR_M}
    markers     = {0: 'o', 1:'^'}
    PLT_SIZE    = 5
    N_ROWS = 1
    N_COLS = 3
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(PLT_SIZE*N_COLS, PLT_SIZE*N_ROWS))
    for i, feature in enumerate(f_labels):
        ax          = axes[i]
        STEP_SIZE   = 10
        MARKER_SIZE = 20
        for diagnosis, colour in colours.items():
            subset = data[data['malignant'] == diagnosis].reset_index()
            if i == 0:
                ax.scatter(
                    subset[features_cv[i]][::STEP_SIZE], subset[features_btl[i]][::STEP_SIZE],
                    s=MARKER_SIZE, marker=markers[diagnosis], color=colour,
                    label=malignancy[diagnosis]
                )
            else:
                ax.scatter(
                    subset[features_cv[i]][::STEP_SIZE], subset[features_btl[i]][::STEP_SIZE],
                    s=MARKER_SIZE, marker=markers[diagnosis], color=colour
                )
            cor_stats = get_rho(data[features_cv[i]], data[features_btl[i]])
            ax.text(
                0.95, 0.1, r"$\rho = $" + f"{np.round(cor_stats['r'],3)}",
                transform=ax.transAxes, fontsize=TEXT_FONT_SIZE, color=FONT_COLOUR,
                verticalalignment='top', horizontalalignment='right', fontproperties=font
            )
            if np.round(cor_stats["p"],3) < 0.001:
                ax.text(
                    0.95, 0.05, "p < 0.001",
                    transform=ax.transAxes, fontsize=TEXT_FONT_SIZE, color=FONT_COLOUR,
                    verticalalignment='top', horizontalalignment='right', fontproperties=font
                )
            else:
                ax.text(
                    0.95, 0.05, f"p = {cor_stats['p']}",
                    transform=ax.transAxes, fontsize=TEXT_FONT_SIZE, color=FONT_COLOUR,
                    verticalalignment='top', horizontalalignment='right', fontproperties=font
                )

        ls = get_ls(data[features_cv[i]], data[features_btl[i]])['lsline']
        ax.plot(data[features_cv[i]], ls, color=FONT_COLOUR, linewidth=2)

        ax.set_title(feature, color=FONT_COLOUR, fontproperties=font, fontsize=FONT_SIZE)
        if i == 1:
            ax.set_xlabel('Computer Vision Estimates',
                          color=FONT_COLOUR, fontproperties=font, fontsize=AXIS_LABEL_FONT_SIZE
                        )
        if i == 0:
            ax.set_ylabel('Perceptual Strength (BTL model)',
                          color=FONT_COLOUR, fontproperties=font, fontsize=AXIS_LABEL_FONT_SIZE
                        )
            ax.legend(loc='upper right', fontsize=TEXT_FONT_SIZE)
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.spines[["top", "right"]].set_visible(False)

    plt.rc_context({'ytick.color':FONT_COLOUR, 'ytick.size':12})
    plt.tight_layout()
    plt.savefig(
        paths['figures'] / 'btl_cv_cor.pdf',
        format='pdf', dpi=600, bbox_inches='tight'
    )

    sns.set_theme(style='ticks')
    mask = np.triu(np.ones_like(cor_features, dtype=bool))
    # it was 'coolwarm', which was actually pretty good.
    cmap_custom = LinearSegmentedColormap.from_list('custom', [COLOUR_B, COLOUR_M])
    print(cor_features)

    N_ROWS = 1
    N_COLS = 1
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(PLT_SIZE*N_COLS, PLT_SIZE*N_ROWS))
    sns.heatmap(cor_features,
            annot=True, fmt=".2f", square=True, linewidths=.5,
            mask=mask,
            xticklabels=['A', 'B', 'C', 'CV_A', 'CV_B', 'CV_C', ''],
            yticklabels=['', 'B', 'C', 'CV_A', 'CV_B', 'CV_C', 'Malignancy'],
            cmap='coolwarm', vmin = 0, vmax = 1, center = 0.5, cbar=False,
            ax=axes
        )
    axes.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "btl_cv_cor_mat.pdf",
        format="pdf", dpi=600, bbox_inches="tight"
    )
    plt.show()
