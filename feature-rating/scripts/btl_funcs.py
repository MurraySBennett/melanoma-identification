from sklearn.linear_model import LogisticRegression
import pandas as pd

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


