import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from matplotlib.font_manager import FontProperties

from plot_funcs import set_style, plt_corr

font = FontProperties(fname="Garamond BoldCondensed.ttf")
font_colour = "#0c2340" # the hex code for contour_colour
font_size = 24

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
# 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42


 ### correlation statistics
    ## sp_rho, sp_p = spearmanr(merged['r'], merged['compact'], nan_policy='omit')
    ## valid_indices = ~np.isnan(merged['r']) & ~np.isnan(merged['compact'])
    ## x = merged['r'][valid_indices]
    ## y = merged['compact'][valid_indices]
    ## p_rho, p_p = pearsonr(x,y)
    ## print(sp_rho, sp_p)
    ## print(p_rho, p_p)

    ### Plotting
    ## colours = ["#6BB8CC", "#C1534B", "#5FAD41", "#9C51B6", "#ED8B00", "#828282"]
    ##https://colorhunt.co/palettes/retro
    ## colours = ['#37E2D5', '#590696', '#C70A80', '#FBCB0A']

    ## rt_fig, rt_ax = plt_rt(regular, irregular, colours)
    ## bias_fig, bias_ax = plt_bias(regular, irregular, colours)
    ## coeff_fig, c_ax = plt_coeffs([ability['q'], ability['r']], colours, labels=['q', 'r'])
    #coeff_fig, c_ax = plt_coeffs(ability['r'], colours)
    #c_ax.set_title("Ranked Ability")
    #c_ax.set_ylabel("Ability")
    #c_ax.set_xlabel("Rank")

    #shape_fig, shape_ax = plt_shape(merged['compact'], colours) 
    #shape_hist_fig, shape_hist_fig = shape_dist(merged, colours, grouped=True)

    #corr_fig, corr_ax = plt_corr(merged['compact'], merged['r'], xlabel='Compact Factor', ylabel='Ability',colours=colours)

    #plt.tight_layout()
    #plt.show()

def get_rho(X, Y, alpha):
    rho, p = spearmanr(X, Y, nan_policy='omit')
    return {'r': rho, 'p': p, 'rp': str(np.round(rho,3))+'*' if p < alpha else str(np.round(rho,3))}

def get_ls(X, Y):
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    # assuming that Y is the BTL data that is currently incomplete
    imputer = SimpleImputer(strategy='mean')
    Y = imputer.fit_transform(Y)
    
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    pred = slope * X + intercept
    return dict(slope=slope, intercept=intercept, lsline=pred)


def equal_scale(x, y):
    min_x, max_x = np.nanmin(x), np.nanmax(x)
    min_y, max_y = np.nanmin(y), np.nanmax(y)
    
    scaled = [(xi - min_x) / (max_x - min_x) * (max_y - min_y) + min_y for xi in x]
    return scaled


def main():
    home_path = path.join(path.expanduser('~'), 'win_home', 'melanoma-identification')
    paths = dict(
        data=path.join(home_path, "feature-rating", "btl-feature-data"),
        figs=path.join(home_path, "feature-rating", "figures")
        )
    data = pd.read_csv(path.join(paths['data'], 'btl-cv-data.csv'))

    # calculate 'combined' symmetry and reverse score compactness for consistent interpretation (higher = bad)
    data['sym'] = data['x_sym'] + data['y_sym']
    data["compact"] = 1-data["compact"]

    colours = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
    set_style(colour_list=colours, fontsize=14)
    features = ['sym', 'compact', 'rms', 'pi_sym', 'pi_bor', 'pi_col']
    f_labels = ['Asymmetry', 'Border Irregularity', 'Colour Variance']
    malignancy = ['Benign', 'Malignant']
    # colours = {0:'#34495E', 1:'#FF6B6B'}  # Navy and Sunset Orange
    colours = {0:'#0c2340', 1:'#D4440D'}  # Navy and Sunset Orange
    # corr_fig, corr_ax = plt_corr(data['x_sym']+data['y_sym'], data['r_sym'], xlabel='CV-Symmetry', ylabel='BTL Strength', colours=colours)
    

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for i, feature in enumerate(f_labels):
        ax = axes[i]
        data[features[i]] = equal_scale(data[features[i]], data[features[i+3]])

        for diagnosis, colour in colours.items():

            subset = data[data['malignant'] == diagnosis].reset_index()
            
            if i == 0:
                ax.scatter(subset[features[i]], subset[features[i+3]], s=4, color=colour, label=malignancy[diagnosis])
                ax.text(0.95, 0.1, f'r = {get_rho(data[features[i]], data[features[i+3]], 0.05)["rp"]}', transform=ax.transAxes,
                    fontsize=14, color=colours[0], verticalalignment='top', horizontalalignment='right', fontproperties=font)
                    # bbox=dict(boxstyle='round, pad=0.3', facecolor='white', alpha=0.8)
                    # )

            else:
                ax.scatter(subset[features[i]], subset[features[i+3]], s=4, color=colour)
                ax.text(0.95, 0.1, f'r = {get_rho(data[features[i]], data[features[i+3]], 0.05)["rp"]}', transform=ax.transAxes,
                    fontsize=14, color=colours[0], verticalalignment='top', horizontalalignment='right', fontproperties=font)
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
    plt.rcParams['text.antialiased'] = True
    # 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
    plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()
    plt.savefig(path.join(paths['figs'], 'btl-cv-cor.pdf'), format='pdf', dpi=800, bbox_inches='tight')
    plt.savefig(path.join(paths['figs'], 'btl-cv-cor.png'), format='png', dpi=800, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    main()

