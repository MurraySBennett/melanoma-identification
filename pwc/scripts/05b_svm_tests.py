""" Compare SVM models and test against chance performance.
Returns:
    _type_: 
"""
from pathlib import Path
from pprint import pprint
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from .cv_transforms import abc_aligned, cv_btl_scale
from ..config import (FILES, PATHS)

def main(bootstrapping=False, test_random=False, test_between=False):
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )
    models = {}
    feature_sets = { "BTL": x[:, 3:], "CV": x[:, :3], "All": x }
    train_feature_sets = {
        "BTL": x_train[:, 3:],
        "CV": x_train[:, :3],
        "All": x_train
    }
    test_feature_sets = {
        "BTL": x_test[:, 3:],
        "CV": x_test[:, :3],
        "All": x_test
    }

    tprs = {}
    preds= {}
    aucs = {}
    rand_aucs = {}
    vs_rand = {}
    bw_results = {}

    for name, features in train_feature_sets.items():
        # best_params = None
        model_params = {
            "kernel": "rbf",
            "C": 0.01,
            "gamma": "auto"
        }
        model = load_model(name, model_params)
        model.fit(features, y_train)
        models[name] = model

    for name, features in test_feature_sets.items():
        model = models[name]
        if bootstrapping:
            start = time.time()
            tprs[name] = bootstrapped_tprs(model, features, y_test)
            save_tprs(tprs[name], name)
            print(f"Saved {name}-tprs in {time.time()-start: .3f} seconds")
        else:
            tprs[name] = load_tprs(name)
        preds[name] = model.predict_proba(features)[:, 1]
        aucs[name] = np.round(roc_auc_score(y_test, preds[name]), 4)

        if test_random:
            print(f"===== Testing {name} against Random =====")
            rand_aucs[name] = permutation_test(
                model,
                feature_sets[name], y,
                n_samples = 100, n_jobs=-1
            )
            vs_rand = np.mean(rand_aucs[name]) >= aucs[name]
            pprint(vs_rand)

    if test_between:
        bw_results["all_cv"] = permutation_test_between(
            y_test,
            preds["All"], preds["CV"],
            label1="All", label2="CV"
        )
        bw_results["all_btl"]= permutation_test_between(
            y_test,
            preds["All"], preds["BTL"],
            label1="All", label2="BTL"
        )
        bw_results["cv_btl"] = permutation_test_between(
            y_test,
            preds["CV"], preds["BTL"],
            label1="CV",  label2="BTL"
        )
        pprint(bw_results)
    pprint(aucs)


config = {
    "paths": {
        "data": FILES['btl_cv'],
        "figures": PATHS['figures'],
        "models": PATHS['svm_models']
    },
    "plotting": {
        "UTSA_BLUE": "#0c2340",
        "UTSA_ORANGE": "#d3a80c",
        "UTSA_COMBINED": "#D04D92",
        "FONT_COLOUR": 'black',
        "FONT_SIZE": 20,
        "AXIS_FONT_SIZE": 18,
        "TEXT_FONT_SIZE": 16,
        "colours": np.divide([
            [230, 159, 0],
            [86, 180, 233],
            [0, 158, 115],
            [204, 121, 167],
            [0, 114, 178],
            ], 255).tolist()
    },
    "feature_labels": ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"],
    "predict_label": ["malignant"]
}

plt.rcParams['text.antialiased'] = True
plt.rcParams['pdf.compression'] = 3
plt.rcParams['pdf.fonttype'] = 42


def get_data():
    """ load data
    Returns:
        dataframe: loaded and partially manipulated data (dropped NAs)
    """
    data = pd.read_csv(config["paths"]["data"])
    data = abc_aligned(data)
    data = cv_btl_scale(data, replace=True)
    data = data[config["feature_labels"] + config["predict_label"]]
    data = data.dropna()
    X = data[config["feature_labels"]]
    y = data[config["predict_label"]].values.ravel()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def bootstrapped_tprs(model, x, y, n_boots=1000, n_jobs=-1):
    mean_fpr = np.linspace(0, 1, 100)
    def bootstrap_iteration(i, random_seed=42):
        rng = np.random.default_rng(random_seed)
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(y[idx])) < 2:
            None
        pred = model.predict_proba(x[idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[idx], pred)
        interp_tpr = np.round(np.interp(mean_fpr, fpr, tpr), 4)
        interp_tpr[0] = 0.0
        print(f"{(i/n_boots)*100:.2f}%")
        return interp_tpr

    seeds = np.random.randint(0, 10000, size=n_boots)
    tprs = Parallel(n_jobs=n_jobs)(delayed(bootstrap_iteration)(i,seeds[i]) for i in range(n_boots))
    tprs = [tpr for tpr in tprs if tpr is not None]
    tprs = np.array(tprs)
    return tprs


def save_tprs(tprs, label):
    df = pd.DataFrame(tprs)
    df.to_csv(config["paths"]["models"] / f"{label}_tprs.csv", index=False)


def load_tprs(label):
    df = pd.read_csv(config["paths"]["models"] / f"{label}_tprs.csv")
    return df


def load_model(name, model_params=None):
    file_name = config["paths"]["models"] / f"{name}_gridSearch.csv"
    df = pd.read_csv(file_name)

    plot_param_scores(df, name)

    if model_params is None:
        model_params = df["params"][0]
    model = SVC(probability=True, **model_params)
    return model
    # return load(config["paths"]["models"] / f"{name}{suffix}_svm.pkl")


def permutation_test(model, x, y, n_samples=100, n_jobs=-1):
    """ permutation test against chance performance
    Tests H0: does model perform better than chance.
    https://stackoverflow.com/questions/52373318/
        how-to-compare-roc-auc-scores-of-different-binary-classifiers-and-assess-statist
    """
    n_splits = 5
    def permutation_iteration(i, seed=42):
        rng = np.random.default_rng(seed)
        auc_values = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(x, y):
            rand_train_idx = train_idx.copy()
            rand_test_idx = test_idx.copy()
            rng.shuffle(rand_train_idx)
            rng.shuffle(rand_test_idx)

            model.fit(x[train_idx], y[rand_train_idx])
            pred = model.predict_proba(x[test_idx])[:, 1]
            auc_values.append(roc_auc_score(y[rand_test_idx], pred))
        print(f"{(i/n_samples)*100:.2f}%")
        return auc_values
    seeds = np.random.randint(0, 10000, size=n_samples)
    results = Parallel(n_jobs=n_jobs)(delayed(permutation_iteration)(i, seed) for i, seed in enumerate(seeds))
    auc_values = [auc for sublist in results for auc in sublist]
    return np.round(np.array(auc_values), 4) 


def permutation_test_between(
    y, pred1, pred2, n_samples = 1000,
    plot = True, label1 = "Model A", label2= "Model B"):
    """ randomly swap predictions between models and see if performance changes.
    Tests the H0 that the two models perform equally well.
    Args:
        y (_type_): _description_
        pred1 (_type_): _description_
        pred2 (_type_): _description_
        n_samples (int, optional): _description_. Defaults to 1000.
    Returns:
        _type_: _description_
    """
    print(f"Running permuation test between {label1} and {label2}.")

    auc_differences = []
    auc1 = roc_auc_score(y, pred1)
    auc2 = roc_auc_score(y, pred2)
    observed_diff = auc1 - auc2
    for _ in range(n_samples):
        mask = np.random.randint(2, size=len(pred1))
        p1 = np.where(mask, pred1, pred2)
        p2 = np.where(mask, pred2, pred1)
        auc1 = roc_auc_score(y, p1)
        auc2 = roc_auc_score(y, p2)
        auc_differences.append(auc1 - auc2)
    auc_differences = np.array(auc_differences)
    results = dict(
        p_value = np.mean(auc_differences >= observed_diff),
        mean_diff = np.mean(auc_differences),
        std_diff = np.std(auc_differences),
        var_diff = np.var(auc_differences),
        ci_95 = np.percentile(auc_differences, [2.5, 97.5])
    )
    if plot:
        fig, ax = plt.subplots(figsize=(6,6))
        nbins = 20
        bin_low, bin_high = -0.06, 0.06
        xlow, xhigh = bin_low, 0.12
        ylow, yhigh = 0, 100
        bins = np.linspace(bin_low, bin_high, nbins)
        ax.hist(auc_differences, bins=bins, alpha=0.7, density=True)
        ax.axvline(
            observed_diff,
            color='black', linestyle='dashed', linewidth=2, alpha=0.8,
            label= "Actual Difference"
        )
        ax.text(
            observed_diff - 0.0025,
            yhigh / 2,
            #max(np.histogram(auc_differences, bins=bins, density=True)[0]/2),
            "Observed Difference",
            color='k', ha='center', va='center', rotation=90
        )
        ax.set_xlabel("Permuted AUC Differences")
        ax.set_ylabel("Density")
        # ax.set_title(f"Permutation dist of AUC diffs bw {labels[0]} and {labels[1]}")
        ax.set_title(f"{label1} vs {label2}")
        ax.set_xlim(xlow, xhigh)
        ax.set_ylim(ylow, yhigh)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.show()

        save_label = f"SVM_AUC_permutation_test_{label1}_{label2}.pdf"
        fig.savefig(config["paths"]["figures"] / save_label, bbox_inches='tight')
    # H0: models perform equally well
    # observed_diff > 0 = model 1 outperforms model 1
    # mean(diffs > obs diffs) the difference is % chance that diff occurred by random chance
    return results


def plot_param_scores(results, model_label):
    mean_test_scores = results["mean_test_score"]
    best_idx = np.argmax(mean_test_scores)
    selected_idx = np.where((results["param_gamma"] == "auto") & (results["param_C"] == 0.01))[0][0]
    plt.figure(figsize=(8, 6))
    ylower = 0.7
    for i, score in enumerate(mean_test_scores):
        params_str = str(results['params'][i])
        colour = 'red' if i == selected_idx else 'black'
        # colour = 'red' if i == best_idx else 'green' if i == selected_idx else 'black'
        plt.plot(i+1, score, 'o', color=colour)
        plt.text(
            i+0.9, ylower + 0.05, params_str,
            fontsize=8, color='black',
            ha='center', va='bottom', rotation=90
        )
    plt.xticks(range(1, len(mean_test_scores)+1))
    plt.xlim(0, len(mean_test_scores)+1)
    plt.ylim(ylower, 0.9)
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('Mean Test Score (ROC AUC)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(
        config["paths"]["figures"] / f"{model_label}_param_search.pdf",
        format='pdf', dpi=600, bbox_inches='tight'
    )


if __name__ == "__main__":
    main(
        bootstrapping=False, # not necessary to run each time. Takes a bit of time, ~5-10 minutes.
        test_random=False, # this takes quite a while so run it once OR just take my scouts-honoured word that they're better than random.
        test_between=True # rapid and informative.
    )
