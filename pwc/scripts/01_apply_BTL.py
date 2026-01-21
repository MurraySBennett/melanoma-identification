from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import (PATHS, FILES)

def main(feature, save_data):

    data_path = PATHS['clean_data']
    estimates = PATHS['estimates']
    if feature is not None:
        for f in feature: 
            if f == "symmetry":
                data = pd.read_csv(data_path / "btl_asymmetry.csv")
            elif f == "border":
                data = pd.read_csv(data_path / "btl_border.csv")
            elif f == "colour":
                data = pd.read_csv(data_path / "btl_colour.csv")

            data = data[data['ended_on'] == 'response'].sort_values(['pID', 'trialNo'])

            print(f"working on {f}")

            X, y = sparse_format(data)
            pi_scores, intercept = lm(X, y, penalty="l2")
            r = pi_scores.to_frame().reset_index()
            r.columns = ['id', 'pi']
            r['pi'] = r['pi'].round(6)

            if save_data:
                r.to_csv(
                    estimates / f"btl_scores_{f}.csv",
                    index = False
                )

    # print('working on ugly')
    # data = pd.read_csv(path.join(data_path, 'data-processed.csv'))
    # X, y = sparse_format(data)
    # r, r_mid, r_slope = lm(X, y, penalty='l2')
    # r = r.to_frame().reset_index().rename(columns={'index': 'id', 0: 'r'})
    # if save_data:
    #     r.to_csv(path.join(data_path, 'btl-scores-global.csv'), index=False)


def sparse_format(data):
    all_images = sorted(list(set(data['img_left']).union(set(data['img_right']))))
    img_to_idx = {img: i for i, img in enumerate(all_images)}

    X = np.zeros((len(data), len(all_images)))
    y = data['response'].values
    
    for i, (_, row) in enumerate(data.iterrows()):
        X[i, img_to_idx[row.img_left]] = -1
        X[i, img_to_idx[row.img_right]] = 1
    return pd.DataFrame(X, columns=all_images), y


def lm(X, y, penalty='l2'):
    model = LogisticRegression(
        penalty=penalty, solver='liblinear', fit_intercept=True, random_state=42, max_iter=1000
    )
    model.fit(X, y)
    coef_series = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    intercept = model.intercept_[0]
    return coef_series, intercept

        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Apply BTL to specified feature data")
    # parser.add_argument("feature", choices=["symmetry", "border", "colour"], help="Feature to process")
    # args = parser.parse_args()
    # main(args.feature)
    features = ["symmetry", "border", "colour"]
    SAVE_DATA = True
    main(features, SAVE_DATA)
