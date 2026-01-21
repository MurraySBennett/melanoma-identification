from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .cv_transforms import abc_aligned, cv_btl_scale
from ..config import (FILES, PATHS)

plt.rcParams.update({
    'pdf.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
})

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculates specific clinical metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return {
        "AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1
    }

def evaluate_model_performance(X_df, y, kf):
    """
    Performs a single pass of CV to collect:
    1. Clinical metrics for tables
    2. ROC curve data + standard deviation for shading
    3. Calibration data
    """
    fold_metrics = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Calibration containers
    mean_prob_true = np.linspace(0, 1, 10)
    list_prob_pred = []

    X = X_df.values
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = SVC(kernel='rbf', C=0.01, gamma="auto", probability=True, random_state=42)
        model.fit(X_train_s, y_train)
        
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = model.predict(X_test_s)
        
        # 1. Metrics
        fold_metrics.append(calculate_metrics(y_test, y_pred, y_prob))
        
        # 2. ROC Data (Interpolate to mean_fpr for averaging)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        
        # 3. Calibration
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        list_prob_pred.append(np.interp(mean_prob_true, prob_pred, prob_true))

    # Aggregate Metrics
    df_m = pd.DataFrame(fold_metrics)
    summary = {col: f"{df_m[col].mean():.3f} (±{df_m[col].std():.2f})" for col in df_m.columns}
    
    # Aggregate ROC (Calculate Mean and Std Dev of TPR)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    # Aggregate Calibration
    mean_cal = np.nanmean(list_prob_pred, axis=0)

    return {
        "summary": summary,
        "roc": {
            "fpr": mean_fpr, 
            "tpr": mean_tpr, 
            "std_tpr": std_tpr, 
            "auc": np.mean(aucs), 
            "std_auc": np.std(aucs)
        },
        "calibration": {"prob_true": mean_cal, "prob_pred": mean_prob_true}
    }

def save_table_figure(results_df, output_path):
    """Renders the dataframe as a clean Matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('off')
    tbl = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        rowLabels=results_df.index,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.get_text().set_weight('bold')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    data = pd.read_csv(FILES["btl_cv"])
    data = abc_aligned(data)
    data = cv_btl_scale(data, replace=True)
    
    feature_labels = ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"]
    y = data["malignant"]

    sets = {
        "Combined (CV + BTL)": data[feature_labels],
        "CV Metrics": data[feature_labels[:3]],
        "BTL Estimates": data[feature_labels[3:]]
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    colours = [[230, 159, 0], [86, 180, 233], [0, 158, 115]]
    colours = np.divide(colours, 255)
    
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(6, 6))
    
    all_summaries = {}

    for i, (name, x_df) in enumerate(sets.items()):
        print(f"Evaluating {name}...")
        results = evaluate_model_performance(x_df, y, kf)
        
        all_summaries[name] = results["summary"]
        
        # Plot ROC with Shaded Error Bars
        r = results["roc"]
        label = f"{name} (AUC = {r['auc']:.2f} ± {r['std_auc']:.2f})"
        ax_roc.plot(r["fpr"], r["tpr"], color=colours[i], lw=2, label=label)
        
        # Shading
        tprs_upper = np.minimum(r["tpr"] + r["std_tpr"], 1)
        tprs_lower = np.maximum(r["tpr"] - r["std_tpr"], 0)
        ax_roc.fill_between(r["fpr"], tprs_lower, tprs_upper, color=colours[i], alpha=0.15)
        
        # Plot Calibration
        c = results["calibration"]
        ax_cal.plot(c["prob_pred"], c["prob_true"], "s-", color=colours[i], 
                    label=f"{name}", markersize=4)

    # Finalize ROC Plot
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curves')
    ax_roc.legend(loc='lower right', frameon=False)
    ax_roc.spines[['top', 'right']].set_visible(False)
    fig_roc.savefig(PATHS["figures"] / "SVM_ROC_Comparison.png", dpi=300, bbox_inches='tight')

    # Finalize Calibration Plot
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    ax_cal.set(xlabel="Mean predicted probability", ylabel="Fraction of positives", title="Calibration Curves")
    ax_cal.legend(loc="lower right", frameon=False)
    ax_cal.spines[['top', 'right']].set_visible(False)
    fig_cal.savefig(PATHS["figures"] / "SVM_Calibration_Comparison.png", dpi=300, bbox_inches='tight')

    # Save Table
    results_df = pd.DataFrame(all_summaries).T
    results_df.to_csv(PATHS["estimates"] / "model_comparison_metrics.csv")
    save_table_figure(results_df, PATHS["figures"] / "model_comparison_table.png")
    
    print(f"Workflow complete. Results saved to {PATHS['figures']}")

if __name__ == "__main__":
    main()
