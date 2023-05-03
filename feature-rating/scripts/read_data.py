import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_style(colour_list, style='plasma', fontsize=12, spines=False):
    """ some settings to practice plotting """
    plt.rcParams['font.size'] = fontsize
    if not spines:
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

    if colour_list is not None:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour_list)

def process_data(file):
    """ read and organise data """
    data_columns = [
    'sender', 'timestamp', 'pID', 
    'condition',  'blockNo', 'practice', 'trialNo', 
    'img_left', 'img_right', 'winner', 'loser', 
    'duration', 'response', 'ended_on']

    df = pd.read_csv(file)

    id_search = re.compile(r'\d+')
    pID = id_search.search(df['url'][0]).group()
    condition = df['condition'][1]
    
    try:
        df = df.loc[(df['sender']=='trial') & (df['practice']==False)]
    except Exception as e:
        print(f'tried to filter {file}, participant {pID}, but {e}')
        return None
    # ignore if data is bad
    try: 
        df['pID'] = pID
        df['condition'] = condition
        df['response'] = df['response'].replace({'nan': np.nan, '0': 0, '1': 1}).astype('Int64')
        df['duration'] = df['duration']
        
    # if df.shape[0] == 0):
        # return None
    # else:
        duplicates = df.duplicated(subset=['blockNo', 'trialNo'], keep=False)
        df = df[~duplicates]
        # df['total_trial']=list(range(1,len(df)+1))
        df = df[data_columns].reset_index(drop=True)
        return df
    except Exception as e:    
        print(f'file {file} is not a complete data set')
        return None


def read_all(files):
    """ read all files into the one dataframe """
    dfs = list(map(process_data, files))
    df = pd.concat(dfs, ignore_index=True)
    grouped_df = df.groupby('pID')
    df['pnum'] = grouped_df.ngroup()
    return df


def plot_rt(reg, irr, colours):
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    # ax[0].axhline(np.mean(summary['exp_duration']), color='r', lw=5, zorder=0)
    ax[0].axhline(np.mean(reg['exp_dur']), c=colours[0], lw=2, linestyle='--')
    ax[0].axhline(np.mean(irr['exp_dur']), c=colours[1], lw=2, linestyle='--')
    ax[0].scatter(reg['pnum'], reg['exp_dur'], c=colours[0], zorder=1, label='reg')
    ax[0].scatter(irr['pnum'], irr['exp_dur'], c=colours[1], zorder=1, label='irr')
    ax[0].legend()
    ax[0].set_title('completion time')

    ax[1].axhline(np.mean(reg['meanRT']), c=colours[0], lw=2, linestyle='--')
    ax[1].axhline(np.mean(irr['meanRT']), c=colours[1], lw=2, linestyle='--')
    ax[1].errorbar(reg['pnum'], reg['meanRT'], reg['semRT'], c=colours[0], fmt='o', capsize=5, zorder=1,  label='reg')
    ax[1].errorbar(irr['pnum'], irr['meanRT'], irr['semRT'], c=colours[1], fmt='o', capsize=5, zorder=1,  label='irr')
    # ax[1].scatter(irr['pnum'], irr['meanRT'], zorder=1, c='g', label='irr')
    # ax[1].legend()
    ax[1].set_title('mean RT')
    

def plot_bias(reg, irr, colours):
    fig, ax = plt.subplots(1, 2, figsize=(4,4))
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


def sem(x):
    return np.std(x) / np.sqrt(len(x))


def meanRT(x):
    return (np.mean(x) - 1500) / 1000


def semRT(x):
    return sem(x) / 1000


def exp_dur(x):
    return np.sum(x) / 1000 / 60


def pos_bias(x):
    return x.dropna().mean()


def pos_rt(x):
    pass


def count_left(x):
    return x.value_counts()[0]


def count_right(x):
    return x.value_counts()[1]


def count_timeouts(x):
    return x.isna().sum()


def main():
    n_files = None
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/feature-rating/"
    paths = dict(
        home=home_path,
        data=os.path.join(home_path, "experiment", "melanoma-2afc", "data"),
        figures=os.path.join(home_path, "figures")
        )
    files=glob.glob(os.path.join(paths['data'], '*.csv'))
    if n_files is not None:
        files = files[:n_files]
    data = read_all(files)
    
    # summary_rt = data.groupby(['condition', 'pnum'])['duration'].agg([
    #     ('meanRT', lambda x: (x.mean() / 1000) - 1.5), 
    #     ('seRT', lambda x: sem(x) / 1000),
    #     ('exp_duration', lambda x: x.sum() / 1000 / 60),
    #     ('n_trials', lambda x: len(x))
    #     ]).reset_index()

    summary = data.groupby(['condition', 'pnum']).agg({
        'response': [pos_bias],
        'duration': [meanRT, semRT, exp_dur]
        }).reset_index()
    # summary.columns = ['{}_{}'.format(col[0], col[1]) if col[1] else col[0] for col in summary.columns]
    summary.columns = [col[1] if col[1] else col[0] for col in summary.columns]

    print(np.unique(data['pID']))
    print(summary)
    # print(summary.mean())

    regular = summary[summary['condition']=='regular']
    irregular = summary[summary['condition']=='irregular']
    
    n_data = len(summary)
    n_reg = len(regular)
    n_irr = len(irregular)
    
    # colours = ["#6BB8CC", "#C1534B", "#5FAD41", "#9C51B6", "#ED8B00", "#828282"]
    #https://colorhunt.co/palettes/retro
    colours = ['#37E2D5', '#590696', '#C70A80', '#FBCB0A']
    set_style(colour_list=colours, fontsize=14)

    plot_rt(regular, irregular, colours)
    plot_bias(regular, irregular, colours)
    plt.show()


if __name__ == '__main__':
    main()
