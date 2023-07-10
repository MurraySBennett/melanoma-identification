import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

def set_style(colour_list, style='plasma', fontsize=12, spines=False):
    """ some settings to practice plotting """
    plt.rcParams['font.size'] = fontsize
    if not spines:
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

    if colour_list is not None:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour_list)


def plt_rt(reg, irr, colours):
    """ plot response time data """
    n_rows, n_cols = 1, 2
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_height,fig_width))

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
    return fig, ax

def plt_bias(reg, irr, colours):
    """ plot response position data """
    n_rows, n_cols = 1, 2
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_height,fig_width))

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
    return fig, ax


def plt_coeffs(coeffs, colours, labels=None):
    """ plot BTL coefficients """
    n_rows, n_cols = 1, 1
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_height,fig_width))

    if labels is not None:
        for counter, y in enumerate(coeffs):
            x = range(len(y))
            y = y.sort_values()
            ax.scatter(x, y, c=colours[counter], s=40, alpha=0.6, label=labels[counter])
    else:
        x = range(len(coeffs))
        y = coeffs.sort_values()
        ax.scatter(x, y, c=colours[0], s=40, alpha=0.6, label='Coeff')
    ax.legend() 
    return fig, ax


def plt_shape(shape, colours, labels=None):
    """ plot x-vision shape factor(s) """
    n_rows, n_cols = 1, 1
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_height,fig_width))

    if labels is not None:
        x = range(len(shape[0]))
        for counter, y in enumerate(shape):
            ax.scatter(x, y, c=colours[counter], s=40, alpha=0.6, label=labels[counter])
    else:
        x = range(len(shape))
        y = shape.sort_values()
        ax.scatter(x, y, c=colours[0], s=40, alpha=0.6, label='Shape')
    ax.legend() 
    return fig, ax


def check_sig(stat, p_value, alpha, direction = 'lt'):
    """ check significance level, and add star for printing """
    if direction == 'lt':
        if p_value < alpha:
            return r"{:.3f}*".format(stat)
        else:
            return r"{:.3f}".format(stat)
    elif direction == 'gt':
        if p_value > alpha:
            return r"{:.3f}*".format(stat)
        else:
            return r"{:.3f}".format(stat)

def get_fig_pos(data, fig_pos):
    """ input array and the position as a proportion of the axis """
    pts = (abs(np.max(data)) + abs(np.min(data)))/100
    return np.min(data) + pts * fig_pos


def plt_corr(x, y, xlabel, ylabel, colours):
    """ plot correlation b/w c-vision shape and BTL """
    sp_rho, sp_p = spearmanr(x, y, nan_policy='omit')
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_indices]
    y = y[valid_indices]
    p_rho, p_p = pearsonr(x,y)
    sp_text = r"$\rho_s = {}$".format(check_sig(sp_rho, sp_p,  0.05))
    p_text = r"$\rho_p = {}$".format(check_sig(p_rho, p_p,  0.05))

    n_rows, n_cols = 1, 1
    fig_width, fig_height = 4*n_rows, 4*n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_height,fig_width))

    ax.scatter(x, y, c=colours[0], s=5, alpha=0.5)
    ax.text(get_fig_pos(x, 10), get_fig_pos(y, 85), p_text, fontsize=12, color='black')
    ax.text(get_fig_pos(x, 10), get_fig_pos(y, 80), sp_text, fontsize=12, color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


