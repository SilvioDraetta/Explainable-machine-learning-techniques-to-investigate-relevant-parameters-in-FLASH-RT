import numpy as np
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

from sklearn.model_selection import cross_validate, StratifiedKFold

def cross_validate_model(model, X_train, y_train, folds=5):
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
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=5)

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


def predict_classes(model, X):
    """
    Generate binary class predictions from a trained Keras model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained neural network model.
    X : numpy.ndarray or pandas.DataFrame
        Input feature matrix.

    Returns
    -------
    numpy.ndarray
        Array of predicted class labels (0 or 1).
    """
    preds = model.predict(X).ravel()
    return (preds > 0.5).astype(int)


def evaluate_keras(model, X, y, evaluator):
    """
    Evaluate a Keras model using an external evaluation function.

    This function generates class predictions using the neural network and then
    passes them to a user-provided evaluation function

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained neural network model.
    X : numpy.ndarray or pandas.DataFrame
        Input features.
    y : numpy.ndarray or pandas.Series
        True labels.
    evaluator : callable
        A function that accepts (y_true, y_pred) and returns evaluation metrics.

    Returns
    -------
    dict
        Dictionary of evaluation metrics returned by the evaluator.
    """
    y_pred = predict_classes(model, X)
    return evaluator(y, y_pred)
