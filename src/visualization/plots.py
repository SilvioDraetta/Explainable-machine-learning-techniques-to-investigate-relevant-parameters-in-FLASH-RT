import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_importance(feature_importances, model_name):
    """
    Plot a horizontal bar chart of feature importances.

    Args:
        feature_importances (list of tuples): (feature_name, importance)
        model_name (str): Name of the model for the plot title.
    """
    features = [f for f, _ in feature_importances]
    values = [v for _, v in feature_importances]

    plt.figure(figsize=(8, 5))
    plt.barh(features, values)
    plt.xlabel("Importance")
    plt.title("Feature Importance - " + model_name)
    plt.gca().invert_yaxis()
    plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, X_val, y_val, model_name):
    """
    Plot the ROC curve for a trained classifier.

    Args:
        model: Trained classifier with predict_proba().
        X_val (pandas.DataFrame): Validation features.
        y_val (pandas.Series): Validation labels.
    """
    probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - " + model_name)
    plt.legend()
    plt.show()

def plot_feature_importance_permutation(df_imp, figsize=(8, 6), model_name="Model"):
    """
    Plot permutation feature importance as a horizontal bar chart.

    Parameters
    ----------
    df_imp : DataFrame
        DataFrame returned by grouped_permutation_importance().
        Must contain 'feature', 'importance_mean', 'importance_std'.
    figsize : tuple, default=(8, 6)
        Size of the matplotlib figure.
    """

    plt.figure(figsize=figsize)
    plt.barh(
        df_imp["feature"],
        df_imp["importance_mean"],
        xerr=df_imp["importance_std"],
        color="steelblue",
        alpha=0.8
    )
    plt.xlabel("Permutation Importance (mean decrease in score)")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importance - " +  model_name)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

def grouped_permutation_importance(model, X, y, titles, folds=5, n_repeats=30, plot=True, model_name="Model"):
    """
    Compute permutation feature importance using stratified group k-fold
    cross-validation. For each fold, the model is trained on the training
    groups and permutation importance is computed on the test groups.

    Parameters
    ----------
    model : estimator
        Any scikit-learn compatible classifier.
    X : DataFrame
        Feature matrix.
    y : Series
        Target labels.
    titles : Series
        Group identifiers (e.g., Title column).
    folds : int, default=5
        Number of cross-validation folds.
    n_repeats : int, default=30
        Number of permutations per feature.
    plot : bool, default=True
        If True, automatically plots the feature importance bar chart.

    Returns
    -------
    DataFrame
        Sorted permutation importances (mean and std across folds).
    """

    groups = titles
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=18)

    all_importances = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=23
        )

        all_importances.append(result.importances_mean)

    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    df_imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": mean_importance,
        "importance_std": std_importance
    }).sort_values("importance_mean", ascending=False)

    if plot:
        plot_feature_importance_permutation(df_imp, model_name=model_name)

    return df_imp




