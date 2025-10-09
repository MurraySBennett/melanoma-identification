from pathlib import Path
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

            data = data[data['ended_on'] == 'response']

            print(f"working on {f}")
            X, y = sparse_format(data)
            r, r_mid, r_slope = lm(X, y, penalty="l2")
            r = r.to_frame().reset_index().rename(columns={"index": "id", 0: "pi"})

            r["pi"] = round(r["pi"], 6)

            print(f"slope: {r_slope}, midpoint: {r_mid}")

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
    rows = []
    for _, row in data.iterrows():
        y = {'y': row.response}
        v = {row.img_left: -1, row.img_right: 1}
        rows.append({**y, **v})
    X = pd.DataFrame(rows)
    X.fillna(0, inplace=True)

    y = X['y']
    X = X.drop(columns=['y'])
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
    # standard errors using inverse of Hessian Matrix
    # decision_function_values = model.decision_function(X)
    # n_samples = X.shape[0]
    # se = np.sqrt(
    #         np.diag(np.linalg.inv(np.dot(X.T * (1- sigmoid(decision_function_values)), X) / n_samples))
            # )
    # Calculate midpoint and slope
    midpoint = -model.intercept_[0] / coefficients[0]
    slope = -coefficients[0] / coefficients[1]
    return q, midpoint, slope#, se

        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Apply BTL to specified feature data")
    # parser.add_argument("feature", choices=["symmetry", "border", "colour"], help="Feature to process")
    # args = parser.parse_args()
    # main(args.feature)
    features = ["symmetry", "border", "colour"]
    SAVE_DATA = True
    main(features, SAVE_DATA)
