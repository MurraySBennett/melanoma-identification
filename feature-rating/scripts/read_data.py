import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from descriptive_funcs import sem, meanRT, semRT, stdRT, exp_dur, pos_bias, count_left, count_right, count_timeouts
from plot_funcs import set_style, plot_rt, plot_bias, plot_coeffs


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

        # df['img_right'] = [i.split('.')[0] for i in df['img_right']]
        # df['winner'] = [i.split('.')[0] for i in df['winner'] if i is not None else np.nan]
        # df['loser'] = [i.split('.')[0] for i in df['loser'] if i is not None else np.nan]

    # if df.shape[0] == 0):
        # return None
    # else:
        duplicates = df.duplicated(subset=['blockNo', 'trialNo'], keep=False)
        df = df[~duplicates]
        # df['total_trial']=list(range(1,len(df)+1))
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
    df['isic_id'] = df['id'].apply(lambda i: i.split('.')[0])
    filtered = df[df['isic_id'].isin(exp_ids)]
    return filtered
    
     
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


def lm(X, y):
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=True)
    l1_model.fit(X, y)
    q = sorted(list(zip(X.columns, l1_model.coef_[0])), key=lambda tup: tup[1], reverse=True)
    q = pd.Series([c for _, c in q], index=[t for t, _ in q])

    l2_model = LogisticRegression(penalty='l2', solver='liblinear', fit_intercept=True)
    l2_model.fit(X, y)
    r = sorted(list(zip(X.columns, l2_model.coef_[0])), key=lambda tup: tup[1], reverse=True)
    r = pd.Series([c for _, c in r], index=[t for t, _ in r])
    
    return q, r


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
    shape_data = pd.read_csv(os.path.join(paths['cv_data'], 'shape.txt'), delim_whitespace=True, header=0)
    shape_data['isic_id'] = [x.strip('.png') for x in shape_data['id']]

    summary = data.groupby(['condition', 'pnum']).agg({
        'response': [pos_bias, count_left, count_right, count_timeouts],
        'duration': [meanRT, semRT, stdRT, exp_dur]
        }).reset_index()
    summary.columns = [col[1] if col[1] else col[0] for col in summary.columns]


    regular = summary[summary['condition']=='regular']
    irregular = summary[summary['condition']=='irregular']
    data = reverse_score(data, 'irregular')

########## error here with winner and loser data.
    ## logistic regression to solve for BTL
    X, y = regression_format(data)
    q, r = lm(X, y)

    # n_data = len(summary)
    # n_reg = len(regular)
    # n_irr = len(irregular)
    
    # colours = ["#6BB8CC", "#C1534B", "#5FAD41", "#9C51B6", "#ED8B00", "#828282"]

    ## Plotting
    #https://colorhunt.co/palettes/retro
    colours = ['#37E2D5', '#590696', '#C70A80', '#FBCB0A']
    set_style(colour_list=colours, fontsize=14)

    plot_rt(regular, irregular, colours)
    plot_bias(regular, irregular, colours)
    plot_coeffs([q, r], colours, labels=['q', 'r'])
    plt.show()
    return {'data': data, 'summary': summary, 'image_ids': image_ids, 'shape_data': shape_data, 'q': q, 'r': r, 'paths': paths}
    
if __name__ == '__main__':
    data = main()
