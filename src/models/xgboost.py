import xgboost as xgb

def train_xgb(X_train, y_train, n=300, md=6, lr=0.05, ss=0.9, csbt=0.9, ra=0.0, rl=1.0):
    """
    Train an XGBoost binary classifier with customizable hyperparameters.

    Parameters
    ----------
    X_train : DataFrame or ndarray
        Training feature matrix.
    y_train : Series or ndarray
        Binary target labels.
    n : int, default=300
        Number of boosting rounds (trees).
    md : int, default=6
        Maximum depth of each tree.
    lr : float, default=0.05
        Learning rate (eta), controls step size shrinkage.
    ss : float, default=0.9
        Subsample ratio for rows.
    csbt : float, default=0.9
        Subsample ratio for columns (per tree).
    ra : float, default=0.0
        L1 regularization term (alpha).
    rl : float, default=1.0
        L2 regularization term (lambda).

    Returns
    -------
    XGBClassifier
        Trained XGBoost model.
    """


    
    model = xgb.XGBClassifier(
        n_estimators=n,
        max_depth=md,
        learning_rate=lr,
        subsample=ss,
        colsample_bytree=csbt,
        reg_alpha=ra,
        reg_lambda=rl,
        random_state=4,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model