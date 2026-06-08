import pandas as pd
import numpy as np


def split_dataset_grouped(df, prezado_count=53, test_fraction=0.25,
                          seed=42, exclude_toschini_group="Ruan et al., 2021"):
    """
    Split the dataset into training and test sets while:
    - keeping all samples of the same paper together,
    - performing manual stratification on Endpoint at the group level,
    - excluding a specific group from TOSCHINI test set,
    - printing detailed information about the split.

    Args:
        df (pd.DataFrame): Full dataset.
        prezado_count (int): Number of rows belonging to PREZADO.
        test_fraction (float): Fraction of each block to include in the test set.
        seed (int): Random seed.
        exclude_toschini_group (str): Group name to always keep in training.

    Returns:
        tuple: (df_train, df_test)
    """

    rng = np.random.RandomState(seed)

    # Split into PREZADO and TOSCHINI blocks
    prezado = df.iloc[:prezado_count].copy()
    toschini = df.iloc[prezado_count:].copy()

    # Exclude the oversized group from TOSCHINI test
    excluded_rows = toschini[toschini["Title"] == exclude_toschini_group]
    toschini = toschini[toschini["Title"] != exclude_toschini_group]

    def manual_group_stratified_split(block, fraction, seed, block_name):
        rng = np.random.RandomState(seed)

        # Group by Title
        groups = block.groupby("Title")

        # Label per group (mode of Endpoint)
        group_labels = groups["Endpoint"].agg(lambda x: x.mode()[0])

        test_groups = []

        # Stratified sampling: sample groups per class
        for endpoint_value in group_labels.unique():
            label_groups = group_labels[group_labels == endpoint_value].index
            n_label_groups = len(label_groups)
            n_test = max(1, int(round(n_label_groups * fraction)))

            chosen = rng.choice(label_groups, size=n_test, replace=False)
            test_groups.extend(chosen)

        df_test_block = block[block["Title"].isin(test_groups)]

        # PRINT INFO
        print(f"\n=== {block_name} TEST SET ===")
        print(f"Selected papers ({len(test_groups)}):")
        for g in test_groups:
            count = (block['Title'] == g).sum()
            print(f"  - {g}  ({count} samples)")
        print(f"Total samples in {block_name} test: {len(df_test_block)}")

        return df_test_block

    # Perform manual stratified sampling
    test_prezado = manual_group_stratified_split(prezado, test_fraction, seed, "PREZADO")
    test_toschini = manual_group_stratified_split(toschini, test_fraction, seed + 1, "TOSCHINI")

    # Combine test set
    df_test = pd.concat([test_prezado, test_toschini], axis=0)

    # Training = everything else (excluded_rows are already in df)
    df_train = df.drop(df_test.index)

    # PRINT GLOBAL SUMMARY
    print("\n=== GLOBAL SPLIT SUMMARY ===")
    print(f"Original dataset size: {len(df)}")
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples:  {len(df_test)}")
    print(f"Excluded group '{exclude_toschini_group}' kept in TRAIN: {len(excluded_rows)} samples")
    print(f"Check: train + test = {len(df_train) + len(df_test)}")

    return df_train, df_test




