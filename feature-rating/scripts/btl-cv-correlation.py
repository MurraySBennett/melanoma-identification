import matplotlib.pyplot as plt
import pandas as pd
from os import path
from scipy.stats import pearsonr, spearmanr
from plot_funcs import set_style, plt_corr

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
    features = ['sym', 'compact', 'rms', 'r_sym', 'r_bor', 'r_col']
    f_labels = ['Asymmetry', 'Border Irregularity', 'Colour Variance']
    malignancy = ['Benign', 'Malignant']
    colours = {0:'#34495E', 1:'#FF6B6B'}  # Navy and Sunset Orange
    # corr_fig, corr_ax = plt_corr(data['x_sym']+data['y_sym'], data['r_sym'], xlabel='CV-Symmetry', ylabel='BTL Strength', colours=colours)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for i, feature in enumerate(f_labels):
        ax = axes[i]
        for diagnosis, colour in colours.items():
            subset = data[data['malignant'] == diagnosis].reset_index()
            if i == 0:
                ax.scatter(subset[features[i]], subset[features[i+3]], s=4, color=colour, label=malignancy[diagnosis])
            else:
                ax.scatter(subset[features[i]], subset[features[i+3]], s=4, color=colour)

        ax.set_title(feature)
        if i == 1:
            ax.set_xlabel('Computer Vision Est.')
        if i == 0:
            ax.set_ylabel('BTL Strength')
            ax.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()

