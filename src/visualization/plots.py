import matplotlib.pyplot as plt

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

