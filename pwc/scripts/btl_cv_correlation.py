import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from matplotlib.font_manager import FontProperties

# from plot_funcs import set_style, plt_corr

font = FontProperties()
font_colour = "black" # the hex code for contour_colour
font_size = 20
axis_label_font_size = 18

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42

def get_rho(X, Y, alpha):
    rho, p = spearmanr(X, Y, nan_policy='omit')
    return {'r': rho, 'p': p, 'rp': str(np.round(rho,3))+'*' if p < alpha else str(np.round(rho,3))}

def get_ls(X, Y):
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    imputer = SimpleImputer(strategy='mean')
    Y = imputer.fit_transform(Y)
    
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    pred = slope * X + intercept
    return dict(slope=slope, intercept=intercept, lsline=pred)


def main():
    f_labels = ['Asymmetry', 'Border Irregularity', 'Colour Variance']
    malignancy = ['Benign', 'Malignant']
    # colours = {0:'#34495E', 1:'#FF6B6B'}  # Navy and Sunset Orange
    colours = {0:'#0c2340', 1:'#D4440D'}  # Navy and Sunset Orange
    markers = {0: 'o', 1:'^'} # circle and triangle

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for i, feature in enumerate(f_labels):
        ax = axes[i]
        step_size = 10
        marker_size = 20 #4
        for diagnosis, colour in colours.items():
            subset = data[data['Diagnosis'] == diagnosis].reset_index()
            
            if i == 0:
                ax.scatter(subset[features[i]][::step_size], subset[features[i+3]][::step_size], s=marker_size, marker=markers[diagnosis], color=colour, label=malignancy[diagnosis])
                ax.text(0.95, 0.1, f'r = {get_rho(data[features[i]], data[features[i+3]], 0.05)["rp"]}', transform=ax.transAxes,
                    fontsize=18, color=colours[0], verticalalignment='top', horizontalalignment='right', fontproperties=font)
                    # bbox=dict(boxstyle='round, pad=0.3', facecolor='white', alpha=0.8)
                    # )

            else:
                ax.scatter(subset[features[i]][::step_size], subset[features[i+3]][::step_size], marker=markers[diagnosis], s=marker_size, color=colour)
                ax.text(0.95, 0.1, f'r = {get_rho(data[features[i]], data[features[i+3]], 0.05)["rp"]}', transform=ax.transAxes,
                    fontsize=18, color=colours[0], verticalalignment='top', horizontalalignment='right', fontproperties=font)
                    # bbox=dict(boxstyle='round, pad=0.3', facecolor='white', alpha=0.8)
                    # )

        ls = get_ls(data[features[i]], data[features[i+3]])['lsline']
        ax.plot(data[features[i]],ls, color=colours[0], linewidth=2)

        ax.set_title(feature, color=font_colour, fontproperties=font, fontsize=font_size)
        if i == 1:
            ax.set_xlabel('Computer Vision Estimates', color=font_colour, fontproperties=font, fontsize=int(font_size*0.75))
        if i == 0:
            ax.set_ylabel('Perceptual Strength (BTL model)', color=font_colour, fontproperties=font, fontsize=int(font_size*0.75))
            ax.legend(loc='upper right')
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])

    plt.rc_context({'ytick.color':font_colour, 'ytick.size':12})
    # plt.rcParams['text.antialiased'] = True
    # 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
    # plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
    # plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()
    # plt.savefig(path.join(paths['figs'], 'btl-cv-cor.pdf'), format='pdf', dpi=600, bbox_inches='tight')
    # plt.savefig(path.join(paths['figs'], 'btl-cv-cor.png'), format='png', dpi=800, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    main()

