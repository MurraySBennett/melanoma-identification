import os
import glob
import re
import builtins
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# from scipy.stats import pearsonr, spearmanr

from descriptive_funcs import sem, meanRT, semRT, stdRT, exp_dur, pos_bias, count_left, count_right, count_timeouts
from plot_funcs import set_style, plt_rt, plt_bias, plt_coeffs, plt_shape, plt_corr


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


    if os.path.getsize(file) < 10_000:
        return None
    
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
        df['winner'] = df['winner'].astype('str')
        df['loser'] = df['loser'].astype('str')
        
        df['winner'] = [x.replace('.JPG', '') for x in df['winner']]
        df['loser'] = [x.replace('.JPG', '') for x in df['loser']]
        df['img_left'] = [x.replace('.JPG', '') for x in df['img_left']]
        df['img_right'] = [x.replace('.JPG', '') for x in df['img_right']]

        duplicates = df.duplicated(subset=['blockNo', 'trialNo'], keep=False)
        df = df[~duplicates]

        df.dropna()
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


def process_shape(path, exp_ids):
    df = pd.read_csv(path, delim_whitespace=True, header=0)
    df['id'] = [x.strip('.png') for x in df['id']]
    df = df[df['id'].isin(exp_ids)].reset_index(drop=True)
    return df
    
     
def regression_format(data):
    def get_vector(r):
        y = {'y': r.response}
        v = {i: 0 for i in images}
        v[r.img_left] = -1
        v[r.img_right] = 1
        return {**y, **v}

    images  = sorted(list(set(data.winner) | set(data.loser)))
    img2idx = {img: idx for idx, img in enumerate(images)}

    X = pd.DataFrame(list(data.apply(get_vector, axis=1)))
    X.fillna(0, inplace=True)
    y = X.y
    X = X[[c for c in X.columns if c != 'y']]

    return X, y


def lm(X, y, penalty='l1'):
    if penalty == 'l1':
        model = LogisticRegression(penalty=penalty, solver='liblinear', fit_intercept=True)
    else:
        model = LogisticRegression(penalty=penalty, solver='liblinear', fit_intercept=True)
    model.fit(X, y)
    q = sorted(list(zip(X.columns, model.coef_[0])), key=lambda tup: tup[1], reverse=True)
    q = pd.Series([c for _, c in q], index=[t for t, _ in q])

    # Extract coefficients
    coefficients = model.coef_[0]
    # Calculate midpoint and slope
    midpoint = -model.intercept_[0] / coefficients[0]
    slope = -coefficients[0] / coefficients[1]

    return q, midpoint, slope


def reverse_score(df, key):
    df.loc[df['condition'] == key, 'response'] = 1 - df.loc[df['condition'] == key, 'response']
    return df


def main():
    n_files = None
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
    paths = dict(
        home=home_path,
        data=os.path.join(home_path, "feature-rating", "experiment", "melanoma-2afc", "data"),
        cv_data=os.path.join(home_path, "computer-vision", "scripts", "feature-analysis"),
        figures=os.path.join(home_path, "feature-rating", "figures")
        )
    files=glob.glob(os.path.join(paths['data'], '*.csv'))
    if n_files is not None:
        files = files[:n_files]
    data = read_all(files)

    image_ids  = sorted(list(set(data.winner) | set(data.loser)))

    ## import shape data
    shape_data = process_shape(os.path.join(paths['cv_data'], 'shape.txt'), image_ids)
    shape_data = shape_data.sort_values('id')
 

    summary = data.groupby(['condition', 'pnum']).agg({
        'response': [pos_bias, count_left, count_right, count_timeouts],
        'duration': [meanRT, semRT, stdRT, exp_dur]
        }).reset_index()
    summary.columns = [col[1] if col[1] else col[0] for col in summary.columns]

    pos_rt = data.groupby(['condition', 'pnum', 'response']).agg({
        'duration': np.mean}).reset_index()

    regular = summary[summary['condition']=='regular']
    irregular = summary[summary['condition']=='irregular']
    data = reverse_score(data, 'irregular')

    ###### error here with winner and loser data.
    ## logistic regression to solve for BTL
    X, y = regression_format(data)
    q, q_mid, q_slope = lm(X, y, penalty='l1')
    r, r_mid, r_slope = lm(X, y, penalty='l2')
    q = q.to_frame().reset_index().rename(columns={'index': 'id', 0: 'q'})
    r = r.to_frame().reset_index().rename(columns={'index': 'id', 0: 'r'})
    ability = pd.merge(q, r, on='id', how='left')
    merged = pd.merge(ability, shape_data, on='id', how='left')
    
    ## correlation statistics
    # sp_rho, sp_p = spearmanr(merged['r'], merged['compact'], nan_policy='omit')
    # valid_indices = ~np.isnan(merged['r']) & ~np.isnan(merged['compact'])
    # x = merged['r'][valid_indices]
    # y = merged['compact'][valid_indices]
    # p_rho, p_p = pearsonr(x,y)
    # print(sp_rho, sp_p)
    # print(p_rho, p_p)

    ## Plotting
    # colours = ["#6BB8CC", "#C1534B", "#5FAD41", "#9C51B6", "#ED8B00", "#828282"]
    #https://colorhunt.co/palettes/retro
    # colours = ['#37E2D5', '#590696', '#C70A80', '#FBCB0A']
    colours = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    set_style(colour_list=colours, fontsize=14)

    rt_fig, rt_ax = plt_rt(regular, irregular, colours)
    bias_fig, bias_ax = plt_bias(regular, irregular, colours)
    coeff_fig, c_ax = plt_coeffs([ability['q'], ability['r']], colours, labels=['q', 'r'])
    c_ax.set_title("abilities")
    shape_fig, shape_ax = plt_shape(merged['compact'], colours) 
    corr_fig, corr_ax = plt_corr(merged['compact'], merged['r'], xlabel='Compactness', ylabel='Ability',colours=colours)

    plt.show()

    return {'data': data, 'summary': summary, 'image_ids': image_ids, 'shape_data': shape_data, 'paths': paths, 'ability': ability, 'merged': merged}
    
if __name__ == '__main__':
    data = main()
