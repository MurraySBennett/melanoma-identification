""" Run the best fitting hyper-parameter SVMs and plot with confidenceintervals derived from
the 5-fold cross-validation stage
Returns:
    _type_: pdf and png plots -- see ../figures/SVM_ROC*
"""
from pathlib import Path
from pprint import pprint
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

from .cv_transforms import abc_aligned, cv_btl_scale
from ..config import (FILES, PATHS)


def main(incl_log=False):
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

    preds= {}
    tprs = {}

    for name, features in train_feature_sets.items():
        model_params = {
            "kernel": "rbf",
            "C": 0.01,
            "gamma": "auto"
        }
        model = load_model(name, model_params)
        model.fit(features, y_train)
        models[name] = model
        preds[name] = model.predict_proba(test_feature_sets[name])[:, 1]
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, _ = roc_curve(y_test, preds[name])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tprs[name] = interp_tpr

    plot_roc_auc(tprs)
    # models = feature_comparison_auc(
    #     df, y, config["feature_labels"]
    # )
    # logreg = None
    # if incl_log:
    #     logreg = stepwise_lr(df[config["feature_labels"]], y)
    # plot_roc_auc(models, y, ["All", "CV", "BTL"], logreg=logreg)


config = {
    "paths": {
        "data": FILES['btl_cv'],
        "figures": PATHS['figures'],
        # "effnet": home.parent / "melnet" / "data",
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
        "colours": {
            "All": np.divide([230, 159, 0], 255),
            "CV": np.divide([86, 180, 233], 255),
            "BTL": np.divide([0, 158, 115], 255),
            "EffNet": np.divide([0, 114, 178], 255),
            "LR": np.divide([213, 94, 0], 255)
        }
    },
    "feature_labels": ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"],
    "predict_label": ["malignant"]
}

plt.rcParams['text.antialiased'] = True
plt.rcParams['pdf.compression'] = 3
plt.rcParams['pdf.fonttype'] = 42


def load_model(name, params=None):
    file_name = config["paths"]["models"] / f"{name}_gridSearch.csv"
    df = pd.read_csv(file_name)
    if params is None:
        params = df["params"][0]
    model = SVC(probability=True, **params)
    return model


def plot_roc_auc(model_tprs, incl_effnet=False, logreg=None):
    mean_fpr = np.linspace(0, 1, 100)
    colours = config["plotting"]["colours"]
    roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
    bench_label = "Random Benchmark"
    rand_bench_line, = roc_ax.plot(
        [0, 1], [0, 1],
        linestyle='--', lw=2, color='black', alpha=.8,
        label=bench_label
    )

    for label, tpr in model_tprs.items():
        boot_data = boot_tprs(label)
        tprs, aucs = boot_data["tprs"], boot_data["aucs"]

        tpr_mean = np.mean(tprs, axis=0)
        tpr_mean.iloc[-1] = 1.0
        tpr_lo, tpr_hi = np.percentile(tprs, [2.5, 97.5], axis=0)

        auc_mean = auc(mean_fpr, tpr_mean)
        auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5], axis=0)

        leg_label = f"{label}:{auc_mean:.2f} ({auc_lo:.2f}, {auc_hi:.2f})"
        if label == "CV":
            leg_label = " " + leg_label
        roc_ax.plot(
            mean_fpr, tpr, #bootstrapped mean is tpr_mean
            label= leg_label,
            lw=2, color=colours[label]
        )
        # plot sem - show estimate precision
        # tpr_sem = np.std(tprs, axis=0) / np.sqrt(len(tprs))
        # roc_ax.fill_between(
        #     mean_fpr,
        #     tpr_mean - tpr_sem, tpr_mean + tpr_sem,
        #     color = colours[label], alpha=0.3
        # )
        ## plot CI - show uncertainty around estimate
        roc_ax.fill_between(
            mean_fpr,
            tpr_lo, tpr_hi,
            color=colours[label], lw=1.2, alpha=0.35
        )

        #     if logreg is not None:
        #         lr_model = logreg["model"]
        #         probas_lr = lr_model.fit(x_train, y_train).predict_proba(x_test)
        #         fpr_lr, tpr_lr, _ = roc_curve(y_test, probas_lr[:, 1])
        #         lr_tprs.append(np.interp(mean_fpr, fpr_lr, tpr_lr))
        #         lr_tprs[-1][0] = 0.0
        #         lr_auc = auc(fpr_lr, tpr_lr)
        #         lr_aucs.append(lr_auc)

    # if logreg is not None:
    #     mean_lr_tpr = np.mean(lr_tprs, axis=0)
    #     mean_lr_tpr[-1] = 1.0
    #     mean_lr_auc = auc(mean_fpr, mean_lr_tpr)
    #     std_lr_tpr = np.std(lr_tprs, axis=0)
    #     roc_ax.plot(
    #         mean_fpr, mean_lr_tpr,
    #         label=f"LR: {mean_lr_auc: .2f}",
    #         lw=2, color=colours[3],
    #     )
    #     lr_tprs_upper = mean_lr_tpr + std_lr_tpr * 1.96 / np.sqrt(len(lr_tprs))
    #     lr_tprs_lower = mean_lr_tpr - std_lr_tpr * 1.96 / np.sqrt(len(lr_tprs))
    #     roc_ax.fill_between(
    #         mean_fpr, lr_tprs_lower, lr_tprs_upper,
    #         color=colours[3], alpha=0.3
    #     )
    roc_ax.set_xlim([0, 1])
    roc_ax.set_ylim([0, 1])
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.spines['top'].set_visible(False)
    roc_ax.spines['right'].set_visible(False)

    handles, labels = roc_ax.get_legend_handles_labels()
    handles = [h for h in handles if h != rand_bench_line]
    labels = [l for l in labels if l != bench_label]
    roc_ax.legend(
        handles, labels,
        loc="lower right",
        title="AUC (95% CI)",
        prop={'family':"DejaVu Sans Mono"}
    )

    save_label = "SVM_ROC"
    if logreg is not None:
        save_label += "_LR"

    pdf_label = f"{save_label}.pdf"
    png_label = f"{save_label}.png"
    roc_fig.savefig(config["paths"]["figures"] / pdf_label, bbox_inches='tight')
    roc_fig.savefig(config["paths"]["figures"] / png_label, bbox_inches='tight', dpi=600)


def stepwise_lr(x, y):
    """ run stepwise logistic regression
    Args:
        x (dataframe): features
        y (list): malignancy
    Returns:
        model dictionary: returns model, fpr, tpr, and roc information for plotting
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    logreg = LogisticRegression()
    rfe = RFECV(logreg, step=1, cv=5)
    rfe = rfe.fit(x_train, y_train)
    selected_ = x.columns[rfe.support_]
    print(f"Optimal number of features: {rfe.n_features_}")
    print(f"Selected features: {selected_}")

    logreg.fit(x_train[selected_], y_train)
    y_pred_prob = logreg.predict_proba(x_test[selected_])[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    y_pred = rfe.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    coefficients = logreg.coef_[0]
    for f, coef in zip(selected_, coefficients):
        # how much more likely lesion is malignant for a one-unit increase in each feature.
        # e.g., if sym = 1.5, then a +1 in symmetry increases the odds of malignancy by 1.5
        print(f"Feature: {f}, Coefficient: {coef: .3f}, odds ratio: {np.exp(coef):.3f}")

    x_train_const = sm.add_constant(x_train[selected_])
    logit_model = sm.Logit(y_train, x_train_const)
    result = logit_model.fit()
    print(result.summary())

    return {'model': logreg, 'fpr': fpr, 'tpr': tpr, 'roc_auc':roc_auc}


def get_data():
    data = pd.read_csv(config["paths"]["data"])
    data = abc_aligned(data)
    data = cv_btl_scale(data, replace=True)
    data = data[config["feature_labels"] + config["predict_label"]]
    data = data.dropna()
    X = data[config["feature_labels"]]
    y = data[config["predict_label"]].values.ravel()
    scaler = StandardScaler().fit(X)
    X = scaler.fit_transform(X)
    return X, y


def boot_tprs(name):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = pd.read_csv(config["paths"]["models"] / f"{name}_tprs.csv")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr.iloc[-1] = 1.0
    aucs = []
    for _, tpr in tprs.iterrows():
        aucs.append(auc(mean_fpr, tpr))
    aucs = np.array(aucs)

    return {
        "tprs": tprs,
        # "tpr_mean": mean_tpr,
        # "tpr_hi" : np.percentile(tprs, 97.5, axis=0),
        # "tpr_lo" : np.percentile(tprs, 2.5, axis=0),
        # "auc_mean": auc(mean_fpr, mean_tpr),
        "aucs": aucs
    }


if __name__ == "__main__":
    main(incl_log=False)
