import xgboost as xgb

def train_xgb(X_train, y_train):
    """
    Train an XGBoost binary classification model on the provided dataset.
    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Feature matrix containing the training samples.
    y_train : pandas.Series or numpy.ndarray
        Target labels for the training samples (binary classification).

    Returns
    -------
    xgboost.XGBClassifier
        A trained XGBoost classifier ready for evaluation or prediction.
    """

    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=4,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model