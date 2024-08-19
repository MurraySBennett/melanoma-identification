from pathlib import Path
# from os import path
import glob
import re
import builtins
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import json
from pprint import pprint as pp

from descriptive_funcs import sem, meanRT, semRT, stdRT, exp_dur, pos_bias, count_left, count_right, count_timeouts
# from plot_funcs import set_style, plt_rt, plt_bias, plt_coeffs, plt_shape, plt_corr, shape_dist

global participant_conditions, pilot_data, sona_ids
participant_conditions = dict(symmetry=0, asymmetry=0,regular=0,irregular=0,uniform=0,colourful=0)
pilot_data = dict(symmetry=0, asymmetry=0,regular=0,irregular=0,uniform=0,colourful=0)
sona_ids = dict(id=[], n_trials=[])
subject = 1

def main():
    save_data   = False
    n_files     = None
    here        = Path(__file__).resolve().parent 
    home_path   = Path(__file__).resolve().parent.parent
    paths = dict(
        data = home_path / "data" / "raw",
        btl_data = home_path / "data" / "cleaned"
        )
    files = list(paths["data"].glob("*elanoma*.csv"))
    if n_files is not None:
        files = files[:n_files]
    data = read_all(files)

    summary = data.groupby(['condition', 'subject']).agg({
        'response': [pos_bias, count_left, count_right, count_timeouts],
        'duration': [meanRT, semRT, stdRT, exp_dur]
        }).reset_index()
    summary.columns = [col[1] if col[1] else col[0] for col in summary.columns]

    if save_data:
        data.to_csv(
            paths["data"] / "00_data-raw.csv",
            index = False
        )

    data = data[data['ended_on'] == 'response']# remove timed-out responses
    data = data[data['duration'] >= 300]
    # reverse scoring conditions -- following this, higher BTL estimates reflect greater irregularity/badness
    data = reverse_score(data, 'symmetry')
    data = reverse_score(data, 'regular') # reverse score regular, so that victories are won by the irregular features. This gives consistency to the other features (A and B), but not the CV estimate (you reverse that, too)
    data = reverse_score(data, 'uniform')

    asymmetry = data[
        (data['condition'] == 'asymmetry') | (data['condition'] == 'symmetry')
        ]
    border = data[
        (data['condition'] == 'irregular') | (data['condition'] == 'regular')
        ]
    colour = data[
        (data['condition'] == 'colourful') | (data['condition'] == 'uniform')
        ]
    return_trials_remaining(data)

    if save_data:
        data.to_csv(
            paths["btl_data"] / "data-processed.csv",
            index=False
        ) # processed == reverse scored
        asymmetry.to_csv(
            paths["btl_data"] / "btl-asymmetry.csv",
            index=False
        )
        border.to_csv(
            paths["btl_data"] / "btl-border.csv",
            index=False
        )
        colour.to_csv(
            paths["btl_data"] / "btl-colour.csv",
            index=False
        )


def process_data(file):
    """ read and organise data """
    global subject
    data_columns = [
    "sender", "timestamp", "pID", 
    "condition",  "blockNo", "practice", "trialNo", 
    "img_left", "img_right", "winner", "loser", 
    "duration", "response", "ended_on"]
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"{file} contains no data")
            return None
    except pd.errors.EmptyDataError:
        print(f"{file} is empty")
        return None
    if file.stat().st_size < 10_000:
        print(f"{file} is an irregularly small file")
        return None

    pID = get_id(df["url"][0], file)
    platform = get_platform(df["url"][0], file)
    if platform is not None and platform == "sona":
        sona_ids["id"].extend([pID])
    condition = df["condition"][1]

    try:
        df = df.loc[(df["sender"] == "trial") & (df["practice"] == False)]
    except Exception as e:
        print(f"{file} has issues filtering out extranous senders: {e}")
        if platform == "sona":
            sona_ids["n_trials"].extend([None])
        return None
    try:
        df["pID"] = pID
        df["condition"] = condition
        df["response"] = df["response"].replace({"nan": np.nan, "0": 0, "1": 1}).astype("Int64")

        df["winner"] = df["winner"].astype("str")
        df["loser"] = df["loser"].astype("str")
        df["winner"] = [x.replace(".JPG", "") for x in df["winner"]]
        df["loser"] = [x.replace(".JPG", "") for x in df["loser"]]

        df["img_left"] = [x.replace(".JPG", "") for x in df["img_left"]]
        df["img_right"] = [x.replace(".JPG", "") for x in df["img_right"]]
        df["duration"] = df["duration"] - 1500 # 1500ms used to load images. The ISI and cue are shown in this initial perdiod.

        duplicates = df.duplicated(subset=["blockNo", "trialNo"], keep = False)
        df = df[~duplicates]
        df = df[~(df["winner"] == "nan")]
        df = df[~(df["loser"] == "nan")]

        df.dropna()
        df = df[data_columns].reset_index(drop=True)
        participant_conditions[condition] += 1
        if platform == "sona":
            sona_ids["n_trials"].extend([len(df)])
        if "btl" in file.stem:
            pilot_data[condition] += 1

        df["subject"] = subject
        subject += 1
        return df

    except Exception as e:
        print(f"{file}: {e}")
        if platform == "sona":
            sona_ids["n_trials"].extend([None])
        return None


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


def get_platform(value, file):
    try:
        data = json.loads(value)
        platform = data.get("platform")
        return platform
    except (json.JSONDecodeError, AttributeError):
        print(f'{file} has no platform information')
        return None



def read_all(files):
    """ read all files into the one dataframe """
    dfs = list(map(process_data, files))
    df = pd.concat(dfs, ignore_index=True)

    print(f"All Data:\n",participant_conditions)
    print(f"Offline Pilot Data:\n", pilot_data)

    pp("SONA IDs:")
    pp(pd.DataFrame(sona_ids).sort_values(by="id"))
    return df


def process_shape(path, exp_ids):
    df = pd.read_csv(path, delim_whitespace=True, header=0)
    df['id'] = [x.strip('.png') for x in df['id']]
    df = df[df['id'].isin(exp_ids)].reset_index(drop=True)
    return df


def reverse_score(df, key):
    df.loc[df['condition'] == key, 'response'] = 1 - df.loc[df['condition'] == key, 'response']
    return df


def return_trials_remaining(data):
    target_trials = 40_500
    participant =  400
    conditions = data['condition'].unique()
    n_participants = 0
    for c in conditions:
        df = data[data['condition'] == c]
        remaining_participants = np.ceil(((target_trials//2) - len(df))/participant)
        print(
            f"{c} has {len(df)}/{target_trials//2} trials. You need {remaining_participants} more participants."
        )
        n_participants += remaining_participants
    print(f'You have a total of {n_participants} remaining')


if __name__ == '__main__':
    main()
