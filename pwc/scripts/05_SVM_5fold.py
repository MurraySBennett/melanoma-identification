from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cv_transforms import abc_aligned, cv_btl_scale

ADD_LR = True

home = Path(__file__).resolve().parent.parent 
paths = {
    "data": home / "data" / "estimates" / "btl_cv_data.csv",
    "figures": home / "figures",
    "scripts": home / "scripts",
    "effnet": home.parent / "melnet" / "data"
}

plt.rcParams['text.antialiased']= True
plt.rcParams['pdf.compression'] = 3
plt.rcParams['pdf.fonttype']    = 42

def main():
    feature_labels  = ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"]
    predict_label   = ["malignant"]

    data = get_data()
    data = data[["id"] + feature_labels + predict_label]
    data = data.dropna(axis=0)
    
    models = feature_comparison_AUC(data, feature_labels)
    roc_auc_plot(models, ["Both", "CV", "BTL"], rev)


def get_data():
    df = pd.read_csv(paths["data"])
    df = abc_aligned(df)
    df = cv_btl_scale(df, replace=True)
    return df


def feature_comparison_AUC(df, f_labels):
    x = df[f_labels].copy()
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    x_cv = df[f_labels[:3]].copy()
    scaler = StandardScaler().fit(x_cv)
    x_cv = scaler.transform(x_cv)

    x_btl = df[f_labels[3:]].copy()
    scaler = StandardScaler().fit(x_btl)
    x_btl = scaler.transform(x_btl)

    y = df["malignant"]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return [(x, y, kf), (x_cv, y, kf), (x_btl, y, kf)]

def cross_val_roc_auc(x, y, kf):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    param_grid = {
        "kernel": "rbf",
        "C": 0.01,
        "gamma": "auto",
        "probability": True
    }
    
    for fold, (train_index, test_index) in enumerate(kf.split(x, y)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = SVC(**param_grid)
        model.fit(x_train, y_train)
        
        y_scores = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    return mean_fpr, mean_tpr, aucs, mean_auc, std_auc, tprs_lower, tprs_upper


def roc_auc_plot(models, plot_labels, revised):
    colours = [[230, 159, 0], [86, 180, 233], [0, 158, 115]]
    colours = np.divide(colours, 255)
    
    roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
    
    for i, (x, y, kf) in enumerate(models):
        mean_fpr, mean_tpr, aucs, mean_auc, std_auc, tprs_lower, tprs_upper = cross_val_roc_auc(x, y, kf)
        
        roc_ax.plot(mean_fpr, mean_tpr, color=colours[i], label=f'{plot_labels[i]}: {mean_auc:.2f} ± {std_auc:.2f}', lw=2)
        roc_ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colours[i], alpha=0.2)
        
    roc_ax.plot([0, 1], [0, 1], 'k--', lw=2)
    roc_ax.set_xlabel('False Positive Rate', fontsize=18)
    roc_ax.set_ylabel('True Positive Rate', fontsize=18)
    roc_ax.legend(loc='lower right', title="AUC", fontsize=16, title_fontsize=18)
    roc_ax.spines['top'].set_visible(False)
    roc_ax.spines['right'].set_visible(False)
    
    save_label = "SVM_ROC"
    pdf_label = f"{save_label}.pdf"
    png_label = f"{save_label}.png"
    
    roc_fig.savefig(paths["figures"] / pdf_label, format="pdf", bbox_inches="tight")
    roc_fig.savefig(paths["figures"] / png_label, format="png", dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    main()
