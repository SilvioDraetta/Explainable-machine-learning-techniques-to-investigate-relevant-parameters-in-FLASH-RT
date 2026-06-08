import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate

def cross_validate_model(model, X_train, y_train, folds=5, seed=8):
    """
    Perform stratified cross-validation on the training set.

    Args:
        model: Classifier to evaluate.
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.
        folds (int): Number of CV folds.

    Returns:
        dict: Mean and std of accuracy and F1 across folds.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"f1": "f1", "accuracy": "accuracy"},
        n_jobs=-1
    )

    return {
        "f1_scores": results["test_f1"],
        "accuracy_scores": results["test_accuracy"],
        "f1_mean": results["test_f1"].mean(),
        "f1_std": results["test_f1"].std(),
        "acc_mean": results["test_accuracy"].mean(),
        "acc_std": results["test_accuracy"].std()
    }


def extract_base_title(title):
    """
    Extract the base title by removing augmentation suffixes.
    
    Examples:
        'paperA_augmented_1' -> 'paperA'
        'paperA_augmented_12' -> 'paperA'
        'paperB' -> 'paperB'
    
    Args:
        title (str): Original title string.
    
    Returns:
        str: Base title without augmentation suffix.
    """
    return re.sub(r'_augmented_\d+$', '', title)


def cross_validate_grouped_stratified(model, X, y, titles, folds=5, seed=8):
    """
    Perform stratified group k-fold cross-validation, keeping all samples
    from the same Title in the same fold to avoid data leakage.

    Groups are defined by the 'titles' vector, and stratification is applied
    at the group level based on the target labels. This ensures that each fold
    has a similar class distribution while preventing samples from the same
    paper from appearing in both train and test splits.

    Parameters
    ----------
    model : estimator
        Any scikit-learn compatible classifier.
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target labels.
    titles : Series
        Group identifiers (e.g., paper titles).
    folds : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    dict
        Cross-validation metrics including per-fold F1 and accuracy scores,
        as well as their mean and standard deviation.
    """

    
    # Extract group identifiers
    groups = titles.apply(extract_base_title)

    # Define the splitter
    sgkf = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed
    )

    print("\n=== Fold composition summary ===\n")

    # Inspect fold composition BEFORE running cross_validate
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        print(f"Fold {fold_idx + 1}:")
        print(f"  Train groups: {len(train_groups)}")
        print(f"  Test groups:  {len(test_groups)}")
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Test samples:  {len(test_idx)}\n")

    # Now run the actual cross-validation
    results = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=sgkf,
        scoring={"f1": "f1", "accuracy": "accuracy"},
        n_jobs=-1
    )

    return {
        "f1_scores": results["test_f1"],
        "accuracy_scores": results["test_accuracy"],
        "f1_mean": results["test_f1"].mean(),
        "f1_std": results["test_f1"].std(),
        "acc_mean": results["test_accuracy"].mean(),
        "acc_std": results["test_accuracy"].std()
    }


