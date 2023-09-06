from os import path
import argparse
import pandas as pd
import numpy as np
from btl_funcs import regression_format, lm

def main(feature, save_data):
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
    data_path = path.join(home_path, "feature-rating", "btl-feature-data")
 
    if feature == "symmetry":
        data = pd.read_csv(path.join(data_path, 'btl-asymmetry.csv'))
    elif feature == "border":
        data = pd.read_csv(path.join(data_path, 'btl-border.csv'))
    elif feature == "colour":
        data = pd.read_csv(path.join(data_path, 'btl-colour.csv'))
    else:
        print("{feature} is invalid input. Please use 'symmetry', 'border', or 'colour'.")
    ## logistic regression to solve for BTL
    X, y = regression_format(data)
    # q, q_mid, q_slope = lm(X, y, penalty='l1')
    #q = q.to_frame().reset_index().rename(columns={'index': 'id', 0: 'q'})
    r, r_mid, r_slope = lm(X, y, penalty='l2')
    r = r.to_frame().reset_index().rename(columns={'index': 'id', 0: 'r'})
    #ability = pd.merge(q, r, on='id', how='left')
    if save_data:
        r.to_csv(path.join(data_path, 'btl-scores-'+feature+'.csv'), index=False)


        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Apply BTL to specified feature data")
    # parser.add_argument("feature", choices=["symmetry", "border", "colour"], help="Feature to process")
    # args = parser.parse_args()
    # main(args.feature)
    feature = 'symmetry'
    save_data = True
    main(feature, save_data)
