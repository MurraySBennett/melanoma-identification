import os
import glob
import re
import pandas as pd
import numpy as np


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
    
    df = df.loc[(df['sender']=='trial') & (df['practice']==False)]


    # ignore if data is bad
    if df.shape[0] == 0:
        return None
    else:
        df['pID'] = pID
        df['condition'] = condition
        
        # you have an error in your experiment script where participants can provide two responses -- delete both until you can fix it
        # duplicates = df.duplicated(subset=['blockNo', 'trialNo'], keep='first')
        duplicates = df.duplicated(subset=['blockNo', 'trialNo'], keep=False)
        df = df[~duplicates]
        # df['total_trial']=list(range(1,len(df)+1))
        df = df[data_columns].reset_index(drop=True)
        return df


def read_all(files):
    """ read all files into the one dataframe """
    dfs = list(map(process_data, files))
    df = pd.concat(dfs, ignore_index=True)
    grouped_df = df.groupby('pID')
    df['pnum'] = grouped_df.ngroup()
    return df

    
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


if __name__ == '__main__':
    main()
