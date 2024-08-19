from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from cv_transforms import abc_aligned, cv_btl_scale
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# plotting variables
UTSA_BLUE       = "#0c2340"
UTSA_ORANGE     = "#d3a80c"
UTSA_COMBINED   = "#D04D92"
FONT_COLOUR     = 'black' 
FONT_SIZE       = 20
AXIS_FONT_SIZE  = 18
TEXT_FONT_SIZE  = 16
# Combined, CV, BTL -- [UTSA_COMBINED, UTSA_ORANGE, UTSA_BLUE]
colours = [
    [230, 159,   0],
    [ 86, 180, 233],
    [  0, 158, 115]
]
colours = np.divide(colours, 255)

plt.rcParams['text.antialiased']= True
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype']    = 42


home = Path(__file__).resolve().parent.parent 
paths = {
    "data": home / "data" / "estimates" / "btl-cv-data.csv"
    "figures": home / "figures",
    "scripts": home / "scripts",
    "effnet": home.parent / "melnet" / "data"
}
data            = pd.read_csv(paths["data"])
data            = abc_aligned(data)
data            = cv_btl_scale(data, replace=True)
feature_labels  = ["sym", "bor", "col", "pi_sym", "pi_bor", "pi_col"]
predict_label   = ["malignant"]
data            = data[["id"] + feature_labels + predict_label]
data            = data.dropna(axis=0)

y = data["malignant"]
X = data[feature_labels]

param_grid = {
    "probability": [True],
    "C": [1],# 10, 50, 100], #10, 50, 100], #[0.01, 0.1, 0.5, 1., 10, 100],
    "kernel": ["rbf"] #,  "poly", "linear"]}
    # "gamma": [0.75, 0.5, 0.25], #[1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
}

def print_score(model, x_train, y_train, x_test, y_test, train=True):
    # compare train and test for over/underfitting
    if train:
        pred = model.predict(x_train)
        report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print(
            f"Training Result:\n\
            Accuracy: {accuracy_score(y_train, pred) *100:.2f}\n\
            Classification Report:\n\
            {report}\n\
            Confusion Matrix:\n\
            {confusion_matrix(y_train, pred)}\n"
        )
    else:
        pred = model.predict(x_test)
        report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        cm = confusion_matrix(y_test, pred)
        TP = cm[0,1]
        TN = cm[1,1]
        FP = cm[0,1] #Type 1 error
        FN = cm[1,0] #Type 2 error
        classification_accuracy = (TP + TN) / float(TP + TN +FP + FN)
        print("==========================================\n")
        print(f"Test Result:\nAccuracy:{accuracy_score(y_test, pred) *100:.2f}\n")
        print("------------------------------------------\n")
        print(f"Classification Accuracy: {classification_accuracy: .2f}")
        print(f"Classification Report:\n{report}\n")
        print("------------------------------------------\n")
        print(f"Confusions:\nTP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}\n")


def test_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
    model.fit(x_train, y_train)
    # print_score(model, x_train, y_train, x_test, y_test, train=True)
    print_score(model, x_train, y_train, x_test, y_test, train=False)


def feature_comparison(df, f_labels):
    x = df[f_labels[:3]]
    print(f"Features: {f_labels[:3]}")
    test_model(x, y)

    x = df[f_labels[3:]]
    print(f"Features: {f_labels[3:]}")
    test_model(x, y)

    x = df[f_labels]
    print(f"Features: {f_labels}")
    test_model(x, y)


def best_parms(model, x_train, y_train, param_grid, refit=True, verbose=1, cv=5):
    grid = GridSearchCV(model, param_grid, refit=refit, verbose=verbose, cv=cv, n_jobs=-1)
    grid.fit(x_train, y_train)
    best_parms = grid.best_params_
    print(best_parms)
    return best_parms


def fit_best(parms, x_train, y_train): #, x_test, y_test):
    model = SVC(**parms)
    model.fit(x_train, y_train)
    # print_score(model, x_train, y_train, x_test, y_test, train=False)
    null_accuracy = y.value_counts()[0] / y.value_counts().sum()
    print(null_accuracy)
    cross_val_roc = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"Cross-validation ROC-AUC: {cross_val_roc:.2f}")
    return model


def feature_comparison_AUC(param_grid, df, f_labels, save_models=False):
    x = df[f_labels].copy()
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    x_cv = df[f_labels[:3]].copy()
    scaler = StandardScaler().fit(x_cv)
    x_cv = scaler.transform(x_cv)

    x_btl = df[f_labels[3:]].copy()
    scaler = StandardScaler().fit(x_btl)
    x_btl= scaler.transform(x_btl)

    if save_models:
        print("================== All features ==================")
        parms = best_parms(SVC(), x, y, param_grid)
        model_all = fit_best(parms, x, y)
        joblib.dump(model_all, paths["scripts"] / "svm_model_combined.pkl")

        print("================== CV features ==================")
        parms = best_parms(SVC(), x_cv, y, param_grid)
        model_cv = fit_best(parms, x_cv, y)
        joblib.dump(model_cv, paths["scripts"] / "svm_model_cv.pkl")

        print("================== BTL features ==================")
        parms = best_parms(SVC(), x_btl, y, param_grid)
        model_btl = fit_best(parms, x_btl, y)
        joblib.dump(model_btl, paths["scripts"] / "svm_model_btl.pkl")

    else:
        model_all   = joblib.load(paths["scripts"] / "svm_model_combined.pkl")
        model_cv    = joblib.load(paths["scripts"] / "svm_model_cv.pkl")
        model_btl   = joblib.load(paths["scripts"] / "svm_model_btl.pkl")
        
    return [(model_all,(x, y)),
            (model_cv, (x_cv, y)),
            (model_btl,(x_btl, y))
            ]


# def plot_training_data_with_decision_boundary(clf, x, y, model_label):
#     # Train the SVC
#     # Settings for plotting
#     fig, ax = plt.subplots(figsize=(4, 3))
#     x_min, x_max, y_min, y_max = -3, 3, -3, 3
#     ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

#     # Plot decision boundary and margins
#     common_params = {"estimator": clf, "X": x, "ax": ax}
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="predict",
#         plot_method="pcolormesh",
#         alpha=0.3,
#     )
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="decision_function",
#         plot_method="contour",
#         levels=[-1, 0, 1],
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#     )

#     # Plot bigger circles around samples that serve as support vectors
#     ax.scatter(
#         clf.support_vectors_[:, 0],
#         clf.support_vectors_[:, 1],
#         s=250,
#         facecolors="none",
#         edgecolors="k",
#     )
#     # Plot samples by color and add legend
#     ax.scatter(X[:, 0], x[:, 1], c=y, s=150, edgecolors="k")
#     ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#     ax.set_title(f"{model_label} Decision boundaries")
#     return fig


def roc_auc(models, plot_labels, include_effnet=False):
    features    = ["CV_A", "CV_B", "CV_C", "A", "B", "C"]
    counter     = 0
    roc_fig, roc_ax = plt.subplots(1,1,figsize=(6,6))
    for model, data in models:
        label = plot_labels[counter]
        if label == "CV + BTL":
            f_labels = np.array(features)
        elif label == "CV":
            f_labels = np.array(features[:3])
        else:
            f_labels = np.array(features[3:])

        x, y = data
        model.fit(x, y)

        fv_fig, fv_ax = plt.subplots(1,1,figsize=(6,6))
        perm_importance = permutation_importance(model, x, y)
        sorted_idx = perm_importance.importances_mean.argsort()
        fv_ax.barh(
            f_labels[sorted_idx], perm_importance.importances_mean[sorted_idx],
            color=UTSA_ORANGE, edgecolor=FONT_COLOUR, linewidth=2, height=0.9
        )
        fv_ax.set_xlabel("Accuracy Loss",
                         color=FONT_COLOUR, fontsize=AXIS_FONT_SIZE)
        fv_ax.set_xlim(left=0, right=0.2)
        fv_ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        fv_ax.tick_params(axis='both', labelsize=AXIS_FONT_SIZE)
        fv_ax.spines['top'].set_visible(False)
        fv_ax.spines['right'].set_visible(False)
        fv_fig.savefig(
            paths["figures"] / f"feature-values-{label}.pdf",
            format='pdf', dpi=600, bbox_inches='tight'
        )

        y_scores = model.predict_proba(x)[:,1]
        fpr, tpr, thresholds = roc_curve(y, y_scores)
        roc_auc = auc(fpr,tpr)

        roc_ax.plot(
            fpr, tpr,
            linewidth=3,  c=colours[counter],
            label=f"{label}: {roc_auc:.2f}"
        )
        # plot decision boundary
        #decision_boundary_fig = plot_training_data_with_decision_boundary(model, X, y, label)
        #decision_boundary_fig.savefig(
            # f"decision_boundary_{label}.pdf",
            # format='pdf', dpi=600, bbox_inches='tight')
        counter += 1

    if include_effnet:
        data_effnet = pd.read_csv(
            paths["effnet"] / 'best_EfficientNetB0-predictions.csv'
        )
        fpr, tpr, _ = roc_curve(data_effnet['malignant'].to_numpy(), data_effnet['prediction'])
        effnet_auc = auc(fpr, tpr)
        roc_ax.plot(
            fpr, tpr,
            linewidth=3, label=f'EN: {effnet_auc:.2f}'
        )

    roc_ax.plot(
        [0, 1], [0, 1],
        'k--', linewidth=2
    )#, label='Random')
    #roc_ax.set_title('SVM Categorisation Comparison',color=FONT_COLOUR, fontsize=FONT_SIZE)
    roc_ax.set_xlabel('False Positive Rate',color=FONT_COLOUR, fontsize=AXIS_FONT_SIZE)
    roc_ax.set_ylabel('True Positive Rate',color=FONT_COLOUR, fontsize=AXIS_FONT_SIZE)
    roc_ax.spines['top'].set_visible(False)
    roc_ax.spines['right'].set_visible(False)
    roc_ax.legend(
        loc='lower right', title="ROC",
        fontsize=TEXT_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE
    )
    roc_fig.savefig(
        paths["figures"] / "SVM_ROC.pdf",
        format="pdf", bbox_inches="tight"
    )
    roc_fig.savefig(
        paths["figures"] / "SVM_ROC.png",
        format="png", dpi=600, bbox_inches="tight"
    )


models = feature_comparison_AUC(param_grid, data, feature_labels, save_models=False)
roc_auc(models, ["CV + BTL", "CV", "BTL"], include_effnet=False)
