from sklearn.ensemble import RandomForestClassifier

def train_random_forest(
    X_train,
    y_train,
    n=300,
    md=None,
    msl=2,
    mss=2,
    cw="balanced",
    rs=4
):
    """
    Train a Random Forest classifier using configurable hyperparameters.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.
        n (int): Number of trees in the forest (n_estimators).
        md (int or None): Maximum depth of each tree.
        msl (int): Minimum number of samples required at a leaf node.
        mss (int): Minimum number of samples required to split an internal node.
        cw (str or dict): Class weight strategy.
        rs (int): Random seed for reproducibility.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=md,
        min_samples_leaf=msl,
        min_samples_split=mss,
        class_weight=cw,
        random_state=rs
    )

    model.fit(X_train, y_train)
    return model
