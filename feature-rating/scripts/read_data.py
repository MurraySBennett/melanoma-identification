from os import path
import glob
import re
import builtins
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import json
# from scipy.stats import pearsonr, spearmanr

from descriptive_funcs import sem, meanRT, semRT, stdRT, exp_dur, pos_bias, count_left, count_right, count_timeouts
from plot_funcs import set_style, plt_rt, plt_bias, plt_coeffs, plt_shape, plt_corr, shape_dist

def get_vars():
    global_vars = {}
    for name, value in globals().items():
        if name not in builtins.__dict__ and not name.startswith('__'):
            var_info = {}
            var_info["type"] = str(value.dtype) if isinstance(value, np.ndarray) else type(value).__name__
            var_info["shape"] = np.shape(value) if isinstance(value, np.ndarray) else None
            var_info["nans"] = np.isnan(value).any() if isinstance(value, np.ndarray) else False
            global_vars[name] = var_info

    return global_vars


def get_id(value, file):
    try:
        data = json.loads(value)
        pID = data.get("participant")
        return pID
    except (json.JSONDecodeError, AttributeError):
        print(f'{file} has ID issues')
        return 'No_ID'


def process_data(file):
    """ read and organise data """
    data_columns = [
    'sender', 'timestamp', 'pID', 
    'condition',  'blockNo', 'practice', 'trialNo', 
    'img_left', 'img_right', 'winner', 'loser', 
    'duration', 'response', 'ended_on']

    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f'{file} contains no data')
            return None
    except pd.errors.EmptyDataError:
        print(f'{file} is empty')
        return None

    # id_search = re.compile(r'\d+')
    # pID = id_search.search(df['url'][0]).group()
    pID = get_id(df['url'][0], file)

    condition = df['condition'][1]

    if path.getsize(file) < 10_000:
        print(f'{file} is an irregularly small file')
        return None
    
    try:
        df = df.loc[(df['sender']=='trial') & (df['practice']==False)]
    except Exception as e:
        print(f'{file} has issues filtering out extranous senders: {e}')
        return None
    try: 
        df['pID'] = pID
        df['condition'] = condition
        df['response'] = df['response'].replace({'nan': np.nan, '0': 0, '1': 1}).astype('Int64')
        df['winner'] = df['winner'].astype('str')
        df['loser'] = df['loser'].astype('str')
        
        df['winner'] = [x.replace('.JPG', '') for x in df['winner']]
        df['loser'] = [x.replace('.JPG', '') for x in df['loser']]
        df['img_left'] = [x.replace('.JPG', '') for x in df['img_left']]
        df['img_right'] = [x.replace('.JPG', '') for x in df['img_right']]
        df['duration'] = df['duration'] - 1500 # 1500ms used to load images. The ISI and cue are shown in this initial perdiod.

        duplicates = df.duplicated(subset=['blockNo', 'trialNo'], keep=False)
        df = df[~duplicates]

        df.dropna()
        df = df[data_columns].reset_index(drop=True)
        return df
    except Exception as e:    
        print(f'{file} has some issue that you need to come back into the code to inspect')
        return None


def read_all(files):
    """ read all files into the one dataframe """
    dfs = list(map(process_data, files))
    df = pd.concat(dfs, ignore_index=True)
    grouped_df = df.groupby('pID')
    df['pnum'] = grouped_df.ngroup()
    return df


def process_shape(path, exp_ids):
    df = pd.read_csv(path, delim_whitespace=True, header=0)
    df['id'] = [x.strip('.png') for x in df['id']]
    df = df[df['id'].isin(exp_ids)].reset_index(drop=True)
    return df
    
     
def reverse_score(df, key):
    df.loc[df['condition'] == key, 'response'] = 1 - df.loc[df['condition'] == key, 'response']
    return df


def main():
    save_data = True
    n_files = None
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
    paths = dict(
        home=home_path,
        data=path.join(home_path, "feature-rating", "experiment", "melanoma-2afc", "data"),
        cv_data=path.join(home_path, "computer-vision", "scripts", "feature-analysis"),
        figures=path.join(home_path, "feature-rating", "figures"),
        mel_id=path.join(home_path, "computer-vision", "scripts", "image_selection"),
        btl_data=path.join(home_path, "feature-rating", "btl-feature-data")
        )
    files=glob.glob(path.join(paths['data'], '*.csv'))
    if n_files is not None:
        files = files[:n_files]
    data = read_all(files)

    # image_ids  = sorted(list(set(data.winner) | set(data.loser)))

    ## import shape data
    # shape_data = process_shape(path.join(paths['cv_data'], 'shape.txt'), image_ids)
    # shape_data = shape_data.sort_values('id')

    ## import melanoma IDs
    # melanoma_ids = pd.read_csv(path.join(paths['mel_id'], 'malignant_ids.txt'))

    summary = data.groupby(['condition', 'pnum']).agg({
        'response': [pos_bias, count_left, count_right, count_timeouts],
        'duration': [meanRT, semRT, stdRT, exp_dur]
        }).reset_index()
    summary.columns = [col[1] if col[1] else col[0] for col in summary.columns]

    pos_rt = data.groupby(['condition', 'pnum', 'response']).agg({
        'duration': np.mean}).reset_index()

    regular = summary[summary['condition']=='regular']
    irregular = summary[summary['condition']=='irregular']
    if save_data:
        data.to_csv(path.join(paths['btl_data'], 'data-raw.csv'), index=False) # raw == no score manipulations -- see data-processed.csv

    data = data[data['ended_on'] == 'response']# remove timed-out responses
    data = data[data['duration'] >= 300]
    data = reverse_score(data, 'symmetry')
    data = reverse_score(data, 'regular') # reverse score regular, so that victories are won by the irregular features. This gives consistency to the other features (A and B), but not the CV estimate (you reverse that, too)
    data = reverse_score(data, 'uniform')

    asymmetry = data[(data['condition'] == 'asymmetry') | (data['condition'] == 'symmetry')]
    border = data[(data['condition'] == 'irregular') | (data['condition'] == 'regular')]
    colour = data[(data['condition'] == 'colourful') | (data['condition'] == 'uniform')]

    if save_data:
        data.to_csv(path.join(paths['btl_data'], 'data-processed.csv'), index=False) # processed == reverse scored
        asymmetry.to_csv(path.join(paths['btl_data'], 'btl-asymmetry.csv'), index=False)
        border.to_csv(path.join(paths['btl_data'], 'btl-border.csv'), index=False)
        colour.to_csv(path.join(paths['btl_data'], 'btl-colour.csv'), index=False)

    ###### error here with winner and loser data.

    ## logistic regression to solve for BTL
    #X, y = regression_format(data)
    #q, q_mid, q_slope = lm(X, y, penalty='l1')
    #r, r_mid, r_slope = lm(X, y, penalty='l2')
    #q = q.to_frame().reset_index().rename(columns={'index': 'id', 0: 'q'})
    #r = r.to_frame().reset_index().rename(columns={'index': 'id', 0: 'r'})
    #ability = pd.merge(q, r, on='id', how='left')
    #merged = pd.merge(ability, shape_data, on='id', how='left')
    #merged = pd.merge(merged, melanoma_ids, on='id', how='left')
    
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
    #colours = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
    #set_style(colour_list=colours, fontsize=14)

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

    #return {'data': data, 'summary': summary, 'image_ids': image_ids, 'shape_data': shape_data, 'paths': paths, 'ability': ability, 'merged': merged}
    
if __name__ == '__main__':
    data = main()
