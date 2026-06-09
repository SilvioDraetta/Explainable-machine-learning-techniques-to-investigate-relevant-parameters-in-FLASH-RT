"""
FUNCTIONS FOR AUGMENTED K-FOLD DATASET CREATION, EVALUATION AND VISUALIZATION
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def augment_dataset_fold(df, num_variants=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    cols_sign_sigma = ["MDR", "PW", "NoP"]

    rows = []

    for _, row in df.iterrows():
        rows.append(row.copy())

        for i in range(1, num_variants + 1):
            new_row = row.copy()

            for col in cols_sign_sigma:
                value = row[col]
                sigma = 0.01 if value < 0 else 0.01
                new_row[col] = value + rng.normal(0, sigma)

            new_row["TD"] = row["TD"] + rng.normal(0, 0.01)
            new_row["Frequency"] = row["Frequency"] + rng.normal(0, 0.01)

            rows.append(new_row)

    return pd.DataFrame(rows).reset_index(drop=True)



def build_augmented_kfold_datasets(
    X, 
    y, 
    num_variants=3,
    n_splits=5,
    random_state=42
):
    """
    Build K-fold datasets where augmentation is applied ONLY to the training split
    of each fold, while the validation split remains untouched.

    This function:
    - Randomly splits the dataset using StratifiedKFold (shuffle=True)
    - For each fold:
        * Takes the training subset
        * Applies the augment_dataset() function
        * Leaves the validation subset unchanged
    - Returns a list of dictionaries, one per fold, containing:
        * X_train (augmented)
        * y_train (augmented)
        * X_val (original)
        * y_val (original)
        * train_idx, val_idx (indices used)

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or array-like): Target vector.
        num_variants (int): Number of augmented copies per training row.
        random_state (int): Seed for reproducibility.

    Returns:
        list[dict]: One dictionary per fold containing augmented training data
                    and untouched validation data.
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    datasets = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        rng = np.random.default_rng(random_state + fold)

        # Training split
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)

        # Validation split
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        # Prepare training df for augmentation
        df_train = X_train.copy()
        df_train["_target_"] = y_train.values

        # AUGMENTATION HAPPENS HERE
        df_train_aug = augment_dataset_fold(
            df_train,
            num_variants=num_variants,
            rng=rng
        )

        # Split back
        y_train_aug = df_train_aug["_target_"].copy()
        X_train_aug = df_train_aug.drop(columns=["_target_"]).reset_index(drop=True)

        datasets.append({
            "fold": fold,
            "X_train": X_train_aug,
            "y_train": y_train_aug,
            "X_val": X_val,
            "y_val": y_val,
            "train_idx": train_idx,
            "val_idx": val_idx
        })

    return datasets

import matplotlib.pyplot as plt

def plot_feature_distributions(X, title="Feature distributions"):
    """
    Plot normalized distributions of all numerical features in a single scatter plot.

    This function:
    - Selects all numerical columns from the input DataFrame.
    - Normalizes each feature using z-score normalization:
        (x - mean) / std
    - Plots each feature as a scatter series with a different color.
    - Allows visual comparison of feature distributions on the same scale.

    Args:
        X (pd.DataFrame): Input dataset containing numerical features.
        title (str): Title of the plot.

    Returns:
        None. Displays a matplotlib scatter plot.
    """
    cols = X.columns.drop(["Particle", "Target"])


    plt.figure(figsize=(8, 6))

    for col in cols:
        plt.scatter(
            np.arange(len(X)),
            X[col],
            s=12,
            alpha=0.6,
            label=col
        )

    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.tight_layout()
    plt.show()


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_classification_folds(datasets, model):
    """
    Evaluate a classification model on multiple folds, computing:
    - Accuracy
    - F1-score (macro)
    - ROC AUC (macro, OVR)

    Additionally stores:
    - y_true and y_proba for each fold (needed for ROC plotting)

    Args:
        datasets (list[dict]): Output of build_augmented_kfold_datasets().
        model: sklearn-like classifier implementing fit() and predict_proba().

    Returns:
        dict: Contains per-fold metrics, aggregated statistics, and fold_results.
    """

    acc_list = []
    f1_list = []
    auc_list = []
    fold_results = []

    for fold_data in datasets:
        X_train = fold_data["X_train"]
        y_train = fold_data["y_train"]
        X_val = fold_data["X_val"]
        y_val = fold_data["y_val"]

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        auc = roc_auc_score(y_val, y_proba[:, 1])

        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)

        # Save fold results for ROC plotting
        fold_results.append({
            "y_true": y_val.values,
            "y_proba": y_proba
        })

        print(f"Fold {fold_data['fold']}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC AUC:  {auc:.4f}")
        print("-" * 40)

    print("\n=== SUMMARY ===")
    print(f"Accuracy: mean={np.mean(acc_list):.4f}, std={np.std(acc_list):.4f}")
    print(f"F1-score: mean={np.mean(f1_list):.4f}, std={np.std(f1_list):.4f}")
    print(f"ROC AUC:  mean={np.mean(auc_list):.4f}, std={np.std(auc_list):.4f}")

    return {
        "accuracy": acc_list,
        "f1": f1_list,
        "auc": auc_list,
        "accuracy_mean": np.mean(acc_list),
        "accuracy_std": np.std(acc_list),
        "f1_mean": np.mean(f1_list),
        "f1_std": np.std(f1_list),
        "auc_mean": np.mean(auc_list),
        "auc_std": np.std(auc_list),
        "fold_results": fold_results
    }

import itertools

def grid_search_classification(
    datasets,
    model_class,
    param_grid,
    scoring="auc"
):
    """
    Perform manual grid search over augmented K-fold datasets.

    For each parameter combination:
    - A new model instance is created
    - The model is trained on the augmented training split of each fold
    - The model is evaluated on the untouched validation split
    - Metrics computed per fold:
        * Accuracy
        * F1-score (macro)
        * ROC AUC (macro, OVR)
    - The metric specified in `scoring` is used to select the best parameters.

    Args:
        datasets (list[dict]):
            Output of build_augmented_kfold_datasets().
            Each dict must contain:
                - X_train, y_train
                - X_val, y_val
        model_class (callable):
            A sklearn-like estimator class (e.g., XGBClassifier, RandomForestClassifier).
            Must be instantiable as model_class(**params).
        param_grid (dict):
            Dictionary of hyperparameters, e.g.:
            {
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.01]
            }
        scoring (str):
            Metric used to select the best parameters.
            Options: "accuracy", "f1", "auc".

    Returns:
        dict:
            {
                "results": list of dicts with metrics for each param combination,
                "best_params": dict,
                "best_scores": dict
            }
    """

    if scoring not in ["accuracy", "f1", "auc"]:
        raise ValueError("scoring must be one of: 'accuracy', 'f1', 'auc'")

    # Generate all combinations of parameters
    keys = list(param_grid.keys())
    param_combinations = list(itertools.product(*param_grid.values()))

    results = []

    for combo in param_combinations:
        params = dict(zip(keys, combo))

        acc_list = []
        f1_list = []
        auc_list = []

        for fold_data in datasets:
            X_train = fold_data["X_train"]
            y_train = fold_data["y_train"]
            X_val = fold_data["X_val"]
            y_val = fold_data["y_val"]

            # Create model with current params
            model = model_class(**params)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Predict probabilities for AUC
            y_proba = model.predict_proba(X_val)

 



  
            

            



            # Metrics
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")
            auc = roc_auc_score(y_val, y_proba[:, 1])

            acc_list.append(acc)
            f1_list.append(f1)
            auc_list.append(auc)

        # Store results for this parameter set
        results.append({
            "params": params,
            "accuracy_mean": np.mean(acc_list),
            "accuracy_std": np.std(acc_list),
            "f1_mean": np.mean(f1_list),
            "f1_std": np.std(f1_list),
            "auc_mean": np.mean(auc_list),
            "auc_std": np.std(auc_list),
        })

    # Select best params based on chosen metric
    metric_key = {
        "accuracy": "accuracy_mean",
        "f1": "f1_mean",
        "auc": "auc_mean"
    }[scoring]

    best = max(results, key=lambda r: r[metric_key])

    return {
        "results": results,
        "best_params": best["params"],
        "best_scores": best
    }


from sklearn.metrics import roc_curve, auc


def plot_roc(results, title="ROC curves for all folds"):
    """
    Plot ROC curves for all folds using the output of evaluate_classification_folds().

    Args:
        results (dict): Output of evaluate_classification_folds().
                        Must contain "fold_results" with:
                            - y_true
                            - y_proba
        title (str): Title of the plot.

    Returns:
        None. Displays a matplotlib plot.
    """

    fold_results = results["fold_results"]

    plt.figure(figsize=(8, 6))

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    for i, fr in enumerate(fold_results):
        y_true = fr["y_true"]
        y_proba = fr["y_proba"]

        # Binary or multiclass
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            # Macro-average ROC
            fpr_dict = {}
            tpr_dict = {}

            for c in range(y_proba.shape[1]):
                fpr_dict[c], tpr_dict[c], _ = roc_curve(y_true == c, y_proba[:, c])

            # Interpolate macro-average
            all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in fpr_dict]))
            mean_tpr = np.zeros_like(all_fpr)
            for c in fpr_dict:
                mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
            mean_tpr /= y_proba.shape[1]

            fpr = all_fpr
            tpr = mean_tpr
            fold_auc = auc(fpr, tpr)

        else:
            # Binary
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            fold_auc = auc(fpr, tpr)

        aucs.append(fold_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        plt.plot(fpr, tpr, lw=1.8, label=f"Fold {i} (AUC = {fold_auc:.3f})")

    # Random classifier
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="red",
        lw=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.3f})"
    )

    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_feature_importances_across_folds(
    model,
    datasets,
    n_repeats=10,
    random_state=42,
    model_name = "model_name"
):
    """
    Compute and plot aggregated feature importances across all folds using:
    1. Model-based feature importance (mean ± std across folds)
    2. Permutation feature importance (mean ± std across folds)

    NOTE: The model is NOT refit. The model passed must already be trained.
    """

    feature_names = datasets[0]["X_train"].columns

    model_importances_list = []
    perm_importances_list = []

    print("\n==============================")
    print(" FEATURE IMPORTANCE PER FOLD ")
    print("==============================\n")

    for fold_idx, fold_data in enumerate(datasets):
        X_val = fold_data["X_val"]
        y_val = fold_data["y_val"]

        # --- Model-based importance ---
        if hasattr(model, "feature_importances_"):
            model_importances = model.feature_importances_
            model_importances_list.append(model_importances)
        else:
            model_importances_list = None

        # --- Permutation importance ---
        perm = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        perm_importances = perm.importances_mean
        perm_importances_list.append(perm_importances)

        # PRINT fold-level values
        print(f"\n--- Fold {fold_idx+1} ---")
        for name, m_imp, p_imp in zip(feature_names, model_importances, perm_importances):
            print(f"{name:15s} | model={m_imp:.5f} | perm={p_imp:.5f}")

    # -----------------------------
    # AGGREGATION
    # -----------------------------
    perm_importances = np.array(perm_importances_list)

    if model_importances_list is not None:
        model_importances = np.array(model_importances_list)
        model_mean = model_importances.mean(axis=0)
        model_std = model_importances.std(axis=0)
    else:
        model_mean = None
        model_std = None

    perm_mean = perm_importances.mean(axis=0)
    perm_std = perm_importances.std(axis=0)

    # -----------------------------
    # PRINT AGGREGATED VALUES
    # -----------------------------
    print("\n==============================================")
    print(" AGGREGATED FEATURE IMPORTANCE (MEAN ± STD) ")
    print("==============================================\n")

    print("MODEL-BASED IMPORTANCE:")
    if model_mean is not None:
        for name, mean, std in zip(feature_names, model_mean, model_std):
            print(f"{name:15s} mean={mean:.5f}   std={std:.5f}")
    else:
        print("Model does not provide feature_importances_")

    print("\nPERMUTATION IMPORTANCE:")
    for name, mean, std in zip(feature_names, perm_mean, perm_std):
        print(f"{name:15s} mean={mean:.5f}   std={std:.5f}")

    # -----------------------------
    # PLOTTING
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Model-based importance ---
    if model_mean is not None:
        idx = np.argsort(model_mean)
        axes[0].barh(
            feature_names[idx],
            model_mean[idx],
            xerr=model_std[idx],
            color="steelblue",
            alpha=0.8
        )
        axes[0].set_title("Model-based Feature Importance (mean ± std)")
    else:
        axes[0].text(0.5, 0.5, "Model does not provide feature_importances_", ha="center")
        axes[0].set_title(f"Model-based Feature Importance -{model_name}")

    # --- Permutation importance ---
    idx_perm = np.argsort(perm_mean)
    axes[1].barh(
        feature_names[idx_perm],
        perm_mean[idx_perm],
        xerr=perm_std[idx_perm],
        color="darkorange",
        alpha=0.8
    )
    axes[1].set_title(f"Permutation Feature Importance (mean ± std) - {model_name}")

    plt.tight_layout()
    plt.show()

    return 

import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def shap_summary_across_folds(model, datasets, max_display=20):
    """
    Compute SHAP feature importance across multiple validation folds
    using a single trained tree-based model.

    Parameters
    ----------
    model : fitted tree model
        RandomForest, XGBoost, LightGBM, etc.

    datasets : list of dict
        Each element must contain:
            {
                "X_val": pd.DataFrame
            }

    max_display : int
        Number of features displayed in SHAP summary plots.

    Returns
    -------
    dict
        {
            "feature_names",
            "shap_per_fold",
            "shap_mean",
            "shap_std"
        }
    """

    feature_names = datasets[0]["X_val"].columns
    shap_importances_list = []

    print("\n==============================")
    print(" SHAP FEATURE IMPORTANCE PER FOLD ")
    print("==============================\n")

    explainer = shap.TreeExplainer(model)

    for fold_idx, fold_data in enumerate(datasets):

        X_val = fold_data["X_val"]
        # ---------------------------------------------------------
        # COMPUTE SHAP VALUES
        # ---------------------------------------------------------
        shap_values = explainer.shap_values(X_val)

        # ---------------------------------------------------------
        # NORMALIZATION OF SHAP OUTPUT
        # ---------------------------------------------------------

        # CASE 1: old SHAP -> list of arrays
        if isinstance(shap_values, list):

            print("SHAP output: list")
            print("Shapes:", [arr.shape for arr in shap_values])

            if len(shap_values) == 2:
                # binary classification
                shap_values = shap_values[1]

            else:
                # multiclass: choose class with largest variance
                shap_values = max(
                    shap_values,
                    key=lambda x: np.var(x)
                )

        # CASE 2: Explanation object
        elif hasattr(shap_values, "values"):

            shap_values = shap_values.values

        # CASE 3: numpy array
        if isinstance(shap_values, np.ndarray):

            print("SHAP array shape:", shap_values.shape)

            if shap_values.ndim == 3:

                # Modern SHAP:
                # (n_samples, n_features, n_classes)
                if (
                    shap_values.shape[0] == X_val.shape[0]
                    and shap_values.shape[1] == X_val.shape[1]
                ):
                    shap_values = shap_values[:, :, 1]

                # Older SHAP:
                # (n_classes, n_samples, n_features)
                elif (
                    shap_values.shape[1] == X_val.shape[0]
                    and shap_values.shape[2] == X_val.shape[1]
                ):
                    shap_values = shap_values[1]

                else:
                    raise ValueError(
                        f"Unrecognized SHAP shape: {shap_values.shape}"
                    )

        # ---------------------------------------------------------
        # REMOVE EXTRA COLUMN IF PRESENT
        # ---------------------------------------------------------

        if shap_values.shape[1] == X_val.shape[1] + 1:

            print(
                "⚠️ Extra SHAP column detected "
                "(expected value) -> removing"
            )

            shap_values = shap_values[:, :-1]

        # ---------------------------------------------------------
        # FINAL CHECK
        # ---------------------------------------------------------

        if shap_values.shape != X_val.shape:

            raise ValueError(
                f"\nShape mismatch\n"
                f"SHAP: {shap_values.shape}\n"
                f"X_val: {X_val.shape}"
            )

        # ---------------------------------------------------------
        # GLOBAL IMPORTANCE
        # ---------------------------------------------------------

        shap_importance = np.abs(shap_values).mean(axis=0)

        shap_importances_list.append(shap_importance)

        # Print fold ranking
        print("\nFeature ranking:")

        ranking = sorted(
            zip(feature_names, shap_importance),
            key=lambda x: x[1],
            reverse=True
        )

        for feat, val in ranking:
            print(f"{feat:20s} {val:.6f}")

        # ---------------------------------------------------------
        # SHAP SUMMARY PLOT
        # ---------------------------------------------------------

        print("\nGenerating SHAP summary plot...")

        shap.summary_plot(
            shap_values,
            X_val,
            max_display=max_display,
            show=True
        )

    # =============================================================
    # AGGREGATION ACROSS FOLDS
    # =============================================================

    shap_importances = np.array(shap_importances_list)

    shap_mean = shap_importances.mean(axis=0)
    shap_std = shap_importances.std(axis=0)

    print("\n====================================")
    print(" SHAP IMPORTANCE (MEAN ± STD)")
    print("====================================\n")

    ranking = sorted(
        zip(feature_names, shap_mean, shap_std),
        key=lambda x: x[1],
        reverse=True
    )

    for feat, mean_val, std_val in ranking:
        print(
            f"{feat:20s} "
            f"mean={mean_val:.6f} "
            f"std={std_val:.6f}"
        )

    # ---------------------------------------------------------
    # AGGREGATED BARPLOT
    # ---------------------------------------------------------

    order = np.argsort(shap_mean)

    plt.figure(figsize=(10, 6))

    plt.barh(
        np.array(feature_names)[order],
        shap_mean[order],
        xerr=shap_std[order]
    )

    plt.xlabel("Mean(|SHAP value|)")
    plt.title("SHAP Feature Importance (mean ± std across folds)")
    plt.tight_layout()
    plt.show()

   
