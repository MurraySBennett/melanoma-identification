from pathlib import Path
# from os import path
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint as pp

import glob
import re
import builtins
import scipy.stats
import matplotlib.pyplot as plt
from pprint import pprint as pp

from ..config import (PATHS, FILES)
from .descriptive_funcs import sem, meanRT, semRT, stdRT, exp_dur, pos_bias, count_left, count_right, count_timeouts
# from plot_funcs import set_style, plt_rt, plt_bias, plt_coeffs, plt_shape, plt_corr, shape_dist

# global participant_conditions, pilot_data, sona_ids
# participant_conditions = dict(symmetry=0, asymmetry=0,regular=0,irregular=0,uniform=0,colourful=0)
# pilot_data = dict(symmetry=0, asymmetry=0,regular=0,irregular=0,uniform=0,colourful=0)
# sona_ids = dict(id=[], n_trials=[])
# subject = 1

def main():
    save_data   = True
    n_files     = None
    all_files = list(PATHS["raw_data"].iterdir())
    files = sorted([
        f for f in all_files if "melanoma" in f.name.lower() and f.suffix == ".csv"
    ])
    if n_files is not None:
        files = files[:n_files]

    context = {
        "subject_counter": 1,
        "participant_conditions": {k: 0 for k in ['symmetry', 'asymmetry', 'regular', 'irregular', 'uniform', 'colourful']},
        "pilot_data": {k: 0 for k in ['symmetry', 'asymmetry', 'regular', 'irregular', 'uniform', 'colourful']},
        "sona_ids": {"id": [], "n_trials": []}
    }

    data = read_all(files, context)
    if data.empty:
        print("No data processed. Exiting.")
        return
    
    summary = data.groupby(['condition', 'subject']).agg({
        'response': [pos_bias, count_left, count_right, count_timeouts],
        'duration': [meanRT, semRT, stdRT, exp_dur]
        }).reset_index()
    summary.columns = [col[1] if col[1] else col[0] for col in summary.columns]

    if save_data:
        data.to_csv( PATHS["raw_data"] / "00_data-raw.csv", index = False)

    data = data[data['ended_on'] == 'response']# remove timed-out responses
    data = data[data['duration'] >= 300]

    # reverse scoring conditions -- following this, higher BTL estimates reflect greater irregularity/badness
    for cond in ['symmetry', 'regular', 'uniform']:
        data = reverse_score(data, cond)

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
            PATHS["clean_data"] / "data_processed.csv",
            index=False
        ) # processed == reverse scored
        asymmetry.to_csv(
            PATHS["clean_data"] / "btl_asymmetry.csv",
            index=False
        )
        border.to_csv(
            PATHS["clean_data"] / "btl_border.csv",
            index=False
        )
        colour.to_csv(
            PATHS["clean_data"] / "btl_colour.csv",
            index=False
        )


def process_data(file, context):
    """ read and organise data """
    data_columns = [
        "sender", "timestamp", "pID", "subject",
        "condition",  "blockNo", "practice", "trialNo", 
        "img_left", "img_right", "winner", "loser", 
        "duration", "response", "ended_on"
    ]

    try:
        if file.stat().st_size < 10_000:
            return None

        df = pd.read_csv(file)
        if df.empty: return None
            # print(f"{file} contains no data")

        first_url = df["url"].dropna().iloc[0] if not df["url"].dropna().empty else "{}"
        pID = get_id(first_url, file)
        platform = get_platform(df["url"][0], file)

        if platform == "sona":
            context["sona_ids"]["id"].append([pID])

        condition = df["condition"].dropna().iloc[0] if "condition" in df.columns else "unknown"

        df = df.loc[(df["sender"] == "trial") & (df["practice"] == False)].copy()
        df["pID"] = pID
        df["condition"] = condition

        for col in ["winner", "loser", "img_left", "img_right"]:
            df[col] = df[col].apply(lambda x: Path(str(x)).stem if pd.notnull(x) else "nan")
        
        df["response"] = pd.to_numeric(df["response"], errors='coerce').astype("Int64")
        df["duration"] = pd.to_numeric(df["duration"], errors='coerce') - 1500

        df = df.drop_duplicates(subset=["blockNo", "trialNo"])
        df = df[df["winner"] != "nan"].dropna(subset=["winner", "loser"])

        context["participant_conditions"][condition] = context["participant_conditions"].get(condition, 0) + 1
        if platform == "sona":
            context["sona_ids"]["n_trials"].append(len(df))
        df["subject"] = context["subject_counter"]
        context["subject_counter"] += 1

        return df[data_columns]
        
    except Exception as e:
        print(f"Error in {file.name}: {e}")
        return None


def read_all(files, context):
    """Aggregates all files into one dataframe"""
    dfs = [process_data(f, context) for f in files]
    valid_dfs = [d for d in dfs if d is not None]
    if not valid_dfs:
        return pd.DataFrame()
    df = pd.concat(valid_dfs, ignore_index=True)
    return df


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


def reverse_score(df, key):
    mask = df['condition'] == key
    if mask.any():
        df.loc[mask, 'response'] = 1 - df.loc[mask, 'response']
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
