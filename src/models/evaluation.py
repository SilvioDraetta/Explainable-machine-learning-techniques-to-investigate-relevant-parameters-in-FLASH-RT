import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

def evaluate_model(model, X_val, y_val):
    """
    Evaluate a trained model on validation data using accuracy, F1-score,
    confusion matrix, and ROC-AUC.

    Args:
        model: Trained classifier.
        X_val (pandas.DataFrame): Validation features.
        y_val (pandas.Series): Validation labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    pred = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    cm = confusion_matrix(y_val, pred)

    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }


def get_feature_importance(model, feature_names):
    """
    Extract feature importance values from a trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        feature_names (list): List of feature names.

    Returns:
        list of tuples: (feature_name, importance) sorted by importance.
    """
    importances = model.feature_importances_
    return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)


def evaluate_model_keras(model, X_val, y_val, threshold=0.5):
    # Probabilità
    probs = model.predict(X_val).ravel()

    # Classi 0/1
    pred = (probs > threshold).astype(int)

    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    cm = confusion_matrix(y_val, pred)

    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }

