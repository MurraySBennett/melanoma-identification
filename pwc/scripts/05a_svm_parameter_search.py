""" Run gridsearch on SVM hyperparameters. Save the parameters and the resulting models
to ../svm_models/ for future use. 
Returns:
    plot: AUC scores for each paramater combination
"""
from pathlib import Path
from joblib import Parallel, delayed, dump
import time
import numpy as np
from sklearn.model_selection import  GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pprint import pprint
import pandas as pd

from cv_transforms import abc_aligned, cv_btl_scale

# gamma:
# 'scale' (default): 1/ (n_features * X.var())
# 'auto': 1/n_features = (1/3, 1/3, 1/6) = (0.33, 0.33, 0.167)
# you might be theoretically better off using 'auto' or 'scale' to account for the 
# different number of features used in each model.

param_grid = {
    "kernel": ["rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
}

def main():
    x, y = get_data()
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, _, y_train, _ = train_test_split(
        x, y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )
    feature_sets = {
        'BTL': x_train[:, 3:],
        'CV': x_train[:, :3],
        'All': x_train
    }
    parallel_results = Parallel(n_jobs=-1)(delayed(perform_grid_search)(name, features, y_train) for name, features in feature_sets.items())
    models = {name: best_estimator for name, best_estimator in parallel_results}
    pprint(models)


home = Path(__file__).resolve().parent.parent
config = {
    "paths": {
        "scripts": home / "scripts",
        "data": home / "data" / "estimates" / "btl_cv_data.csv",
        "figures": home / "figures",
        "effnet": home.parent / "melnet" / "data",
        "models": home / "models"
    },
    "feature_labels": ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"],
    "predict_label": ["malignant"]
}


def get_data():
    """ load data according to whether we're using revised BTL data or not.
    Returns:
        _type_: _description_
    """
    data = pd.read_csv(config["paths"]["data"])
    data = abc_aligned(data)
    data = cv_btl_scale(data, replace=True)
    data = data[config["feature_labels"] + config["predict_label"]]
    data = data.dropna()
    X = data[config["feature_labels"]]
    y = data[config["predict_label"]].values.ravel()
    return X, y


def perform_grid_search(name, features, y):
    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=5,
        scoring = 'roc_auc'
    )
    start = time.time()
    grid_search.fit(features, y)
    bestimator = grid_search.best_estimator_
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by='mean_test_score', ascending=False)
    save_grid_results(results, name)
    # no need, now - I select and fit the model in the next script from the saved grid results.
    # save_model(bestimator, name)
    print(f"{time.time() - start:.2f}s to search {name} parameters.")
    return name, bestimator


def save_grid_results(results, label):
    results.to_csv(
        config["paths"]["models"] / f"{label}_gridSearch.csv",
        index=False
    )


def save_model(model, label):
    dump(model, config["paths"]["models"] / f"{label}_svm.pkl")


if __name__ == "__main__":
    main()
