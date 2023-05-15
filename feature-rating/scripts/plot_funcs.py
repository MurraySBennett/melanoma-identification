import matplotlib.pyplot as plt
import numpy as np

def set_style(colour_list, style='plasma', fontsize=12, spines=False):
    """ some settings to practice plotting """
    plt.rcParams['font.size'] = fontsize
    if not spines:
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

    if colour_list is not None:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour_list)


def plot_rt(reg, irr, colours):
    """ plot response time data """
    n_rows, n_cols = 1, 2
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width*n_cols,fig_height*n_rows))

    ax[0].axhline(0, c=colours[3], lw=2, linestyle='-', zorder=0)
    ax[0].axhline(np.mean(reg['meanRT']), c=colours[0], lw=2, linestyle='--')
    ax[0].axhline(np.mean(irr['meanRT']), c=colours[1], lw=2, linestyle='--')
    ax[0].errorbar(reg['pnum'], reg['meanRT'], reg['stdRT'], c=colours[0], fmt='o', capsize=5, zorder=1,  label='reg')
    ax[0].errorbar(irr['pnum'], irr['meanRT'], irr['stdRT'], c=colours[1], fmt='o', capsize=5, zorder=1,  label='irr')
    # ax[0].scatter(irr['pnum'], irr['meanRT'], zorder=1, c='g', label='irr')
    # ax[0].legend()
    ax[0].set_title('mean RT')
    
    # ax[1].axhline(np.mean(summary['exp_duration']), color='r', lw=5, zorder=0)
    ax[1].axhline(np.mean(reg['exp_dur']), c=colours[0], lw=2, linestyle='--')
    ax[1].axhline(np.mean(irr['exp_dur']), c=colours[1], lw=2, linestyle='--')
    ax[1].scatter(reg['pnum'], reg['exp_dur'], c=colours[0], zorder=1, label='reg')
    ax[1].scatter(irr['pnum'], irr['exp_dur'], c=colours[1], zorder=1, label='irr')
    ax[1].legend()
    ax[1].set_title('completion time')


def plot_bias(reg, irr, colours):
    """ plot response position data """
    n_rows, n_cols = 1, 2
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width*n_cols,fig_height*n_rows))

    ax[0].axhline(np.mean(reg['pos_bias']), c=colours[0], lw=2, linestyle='--')
    ax[0].axhline(np.mean(irr['pos_bias']), c=colours[1], lw=2, linestyle='--')
    ax[0].scatter(reg['pnum'], reg['pos_bias'], c=colours[0], zorder=1, label='reg')
    ax[0].scatter(irr['pnum'], irr['pos_bias'], c=colours[1], zorder=1, label='irr')
    ax[0].legend()
    ax[0].set_title('position bias')
    
    # plot rt for left and right positions
    # ax[1].axhline(np.mean(reg['pos_bias']), lw=2, linestyle='--')
    # ax[1].axhline(np.mean(irr['pos_bias']), lw=2, linestyle='--')
    # ax[1].scatter(reg['pnum'], reg['pos_bias'], zorder=1, label='reg')
    # ax[1].scatter(irr['pnum'], irr['pos_bias'], zorder=1, label='irr')
    # ax[1].legend()
    ax[1].set_title('position RT')


def plot_coeffs(coeffs, colours, labels=None):
    """ plot BTL coefficients """
    n_rows, n_cols = 1, 1
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width*n_cols,fig_height*n_rows))

    if labels is not None:
        x = range(len(coeffs[0]))
        ax.scatter(x, coeffs[0], c=colours[0], s=40, alpha=0.6, label=labels[0])
        ax.scatter(x, coeffs[1], c=colours[1], s=40, alpha=0.6, label=labels[1])
    else:
        x = range(len(coeffs))
        ax.scatter(x, coeffs, c=colours[0], s=40, alpha=0.6, label=labels)
    ax.legend() 


def plt_shape(shape, colours, labels=None):
    """ plot x-vision shape factor(s) """
    n_rows, n_cols = 1, 1
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width*n_cols,fig_height*n_rows))

    if labels is not None:
        x = range(len(shape[0]))
        counter = 0
        for s in shape:
            ax.scatter(x, s, c=colours[counter], s=40, alpha=0.6, label=labels[counter])
            counter += 1
    else:
        x = range(len(shape))
        ax.scatter(x, shape, c=colours[0], s=40, alpha=0.6, label=labels)
    ax.legend() 

def plt_shape_coeffs(shape, coeffs, colours):
    """ plot correlation b/w c-vision shape and BTL """
    n_rows, n_cols = 1, 1
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width*n_cols,fig_height*n_rows))

    shape = shape.sort_values('id')
    coeffs = coeffs.sort_values('isic_id')
    x = shape['compact']
    y = coeffs['coeff']
    ax.scatter(x, y, c=colours[0], s=40, alpha=0.8)
    ax.set_xlabel('Shape Factor')
    ax.set_ylabel('Strength Coeff')

