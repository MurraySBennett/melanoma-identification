import os
import glob
import pandas as pd
import numpy as np

def set_paths(home_path):
    """ takes the home path for the feature-rating directory and returns
    a dictionary with the home, data, and script paths.
    """
    paths = dict(
        home=home_path,
        data=os.path.join(home_path, "btl-simulation", "simulation-data"),
        scripts=os.path.join(home_path, "scripts")
    )
    return paths

def read_data(paths):
    """ Take paths dictionary to read csv data file from data directory. This assumes you have a single data file.
    """
    files = glob.glob(paths["data"] + "/" + "*.csv")
    df = pd.read_csv(files[0])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

