from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(feature, save_data):
    home = Path(__file__).resolve().parent.parent
    data_path = home / "data" / "cleaned"
    estimates = home / "data" / "estimates"
    if feature is not None:
        for f in feature: 
            if f == "symmetry":
                data = pd.read_csv(data_path / "btl-asymmetry.csv")
            elif f == "border":
                data = pd.read_csv(data_path / "btl-border.csv")
            elif f == "colour":
                data = pd.read_csv(data_path / "btl-colour.csv")
            print(f"working on {f}")

            X, y = sparse_format(data)
            r, r_mid, r_slope = lm(X, y, penalty="l2")
            r = r.to_frame().reset_index().rename(columns={"index": "id", 0: "pi"})
            print(f"slope: {r_slope}, midpoint: {r_mid}")

            if save_data:
                r.to_csv(
                    estimates / f"btl-scores-{f}.csv",
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
    feature = ["symmetry", "border", "colour"]
    save_data = True
    main(feature, save_data)


# https://www.google.com/search?q=perceptual+rating+of+melanoma+features+human+perception+judgment&sca_esv=564367827&ei=vTL_ZOfWBY_W5NoP1P-wuAU&ved=0ahUKEwinqdef8KKBAxUPK1kFHdQ_DFcQ4dUDCBA&uact=5&oq=perceptual+rating+of+melanoma+features+human+perception+judgment&gs_lp=Egxnd3Mtd2l6LXNlcnAiQHBlcmNlcHR1YWwgcmF0aW5nIG9mIG1lbGFub21hIGZlYXR1cmVzIGh1bWFuIHBlcmNlcHRpb24ganVkZ21lbnRIxEBQkRFYzDVwB3gBkAEBmAHCAaAB-xeqAQQ2LjIxuAEDyAEA-AEBwgIKEAAYRxjWBBiwA8ICCBAAGIkFGKIEwgIFEAAYogTiAwQYACBBiAYBkAYI&sclient=gws-wiz-serp
# https://www.mdpi.com/2075-1729/13/4/974
# https://academic.oup.com/milmed/article/185/3-4/506/5607587
# https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=human+judgment+melanoma+features+costly+time+and+financial&btnG=
