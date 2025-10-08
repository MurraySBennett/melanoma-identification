from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from joblib import Parallel, delayed

from cv_transforms import abc_aligned, cv_btl_scale

ADD_LR = True

home = Path(__file__).resolve().parent.parent
paths = {
    "data": home / "data" / "estimates" / "btl_cv_data.csv",
    "figures": home / "figures",
    "scripts": home / "scripts",
    "effnet": home.parent / "melnet" / "data"
}


plt.rcParams['text.antialiased'] = True
plt.rcParams['pdf.compression'] = 3
plt.rcParams['pdf.fonttype'] = 42

def main():
    feature_labels = ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"]
    predict_label = ["malignant"]

    data = get_data()
    data = data[["id"] + feature_labels + predict_label]
    data = data.dropna(axis=0)

    all_predictions, y, kf = feature_comparison_AUC(data, feature_labels)

    # Average predictions across folds
    averaged_predictions = {}
    for name, fold_preds in all_predictions.items():
        all_fold_preds = []
        test_index_list = []
        for train_index, test_index in kf.split(data[feature_labels], y):
            test_index_list.append(test_index)
            all_fold_preds.append(fold_preds.pop(0))
        all_fold_preds = np.concatenate(all_fold_preds)
        test_index_list = np.concatenate(test_index_list)
        averaged_predictions[name] = pd.Series(all_fold_preds, index=test_index_list).sort_index().values

    # Permutation tests
    results_cv_vs_both = permutation_test_between(y, averaged_predictions["CV"], averaged_predictions["Both"], label1="CV", label2="Both")
    results_btl_vs_both = permutation_test_between(y, averaged_predictions["BTL"], averaged_predictions["Both"], label1="BTL", label2="Both")
    results_btl_vs_cv = permutation_test_between(y, averaged_predictions["BTL"], averaged_predictions["CV"], label1="BTL", label2="CV")

    print("=== Results (CV vs Both) ===")
    pprint(results_cv_vs_both)
    print("=== Results (BTL vs Both) ===")
    pprint(results_btl_vs_both)
    print("=== Results (BTL vs CV) ===")
    pprint(results_btl_vs_cv)
    return(dict(cv_both=results_cv_vs_both, btl_both=results_btl_vs_both, btl_cv=results_btl_vs_cv))

def get_data():
    df = pd.read_csv(paths["data"])
    df = abc_aligned(df)
    df = cv_btl_scale(df, replace=True)
    return df


def feature_comparison_AUC(df, f_labels, n_jobs=-1):  # Added n_jobs parameter
    x = df[f_labels].copy()
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    x_cv = df[f_labels[:3]].copy()
    scaler = StandardScaler().fit(x_cv)
    x_cv = scaler.transform(x_cv)

    x_btl = df[f_labels[3:]].copy()
    scaler = StandardScaler().fit(x_btl)
    x_btl = scaler.transform(x_btl)

    y = df["malignant"].values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_predictions = {}
    for name, data in [("Both", x), ("CV", x_cv), ("BTL", x_btl)]:
        fold_predictions = Parallel(n_jobs=n_jobs)(
            delayed(train_and_predict)(data[train_index], data[test_index], y[train_index], y[test_index])
            for train_index, test_index in kf.split(data, y)
        )
        all_predictions[name] = fold_predictions

    return all_predictions, y, kf


def train_and_predict(x_train, x_test, y_train, y_test):
    model = SVC(kernel="rbf", C=0.01, gamma="auto", probability=True)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def permutation_test_between(y, pred1, pred2, n_samples=1000, plot=True, label1="Model A", label2="Model B"):
    """
    Randomly swap predictions between models and see if performance changes.
    H0: predictions between models are the same -- 'hoping' for values < 0.05
    """
    
    print(f"Running permutation test between {label1} and {label2}.")

    auc_differences = []
    auc1_observed = roc_auc_score(y, pred1)
    auc2_observed = roc_auc_score(y, pred2)
    observed_diff = auc1_observed - auc2_observed

    for _ in range(n_samples):
        mask = np.random.randint(2, size=len(pred1))
        p1 = np.where(mask, pred1, pred2)
        p2 = np.where(mask, pred2, pred1)

        auc1_permuted = roc_auc_score(y, p1)
        auc2_permuted = roc_auc_score(y, p2)
        auc_diff_permuted = auc1_permuted - auc2_permuted
        auc_differences.append(auc_diff_permuted)

    auc_differences = np.array(auc_differences)

    p_value = np.mean(np.abs(auc_differences) >= np.abs(observed_diff))

    results = dict(
        p_value=p_value,
        mean_diff=np.mean(auc_differences),
        std_diff=np.std(auc_differences),
        var_diff=np.var(auc_differences),
        ci_95=np.percentile(auc_differences, [2.5, 97.5])
    )

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        nbins = 20
        bin_low, bin_high = min(min(auc_differences), observed_diff) - .01 , max(max(auc_differences), observed_diff) + .01
        bins = np.linspace(bin_low, bin_high, nbins)
        ax.hist(auc_differences, bins=bins, alpha=0.7, density=True)

        ax.axvline(observed_diff, color='black', linestyle='dashed', linewidth=2, alpha=0.8, label="Actual Difference")
        ax.text(observed_diff-0.005, ax.get_ylim()[1] / 2, "Observed Difference", color='k', ha='center', va='center', rotation=90)
        ax.text(0.95, 0.95, f"p-value: {p_value:.3f}", transform=ax.transAxes, ha='right', va='top')

        ax.set_xlabel("Permuted AUC Differences")
        ax.set_ylabel("Density")
        ax.set_title(f"{label1} vs {label2}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        save_label = f"SVM_AUC_permutation_test_{label1}_{label2}.pdf"
        fig.savefig(paths["figures"] / save_label, bbox_inches='tight')
    return results

if __name__ == '__main__':
    results = main()
