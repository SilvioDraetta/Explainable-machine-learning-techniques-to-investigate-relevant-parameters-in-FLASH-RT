import numpy as np
import pandas as pd

def augment_dataset(df, num_variants=3):
    """
    Generate augmented versions of each row in the dataset by adding Gaussian
    noise to selected numerical features while keeping fixed columns unchanged.

    Args:
        df (pandas.DataFrame): Training dataset to augment.
        num_variants (int): Number of augmented copies to generate per row.

    Returns:
        pandas.DataFrame: Dataset containing original and augmented rows.
    """
    fixed_cols = ["Particle", "Target", "Endpoint", "Title"]
    cols_sign_sigma = ["MDR", "PW", "NoP"]

    final_rows = []

    for _, row in df.iterrows():
        # Add original row
        final_rows.append(row.copy())

        original_title = row["Title"]

        # Generate augmented variants
        for i in range(1, num_variants + 1):
            new_row = row.copy()

            # Modify title
            new_row["Title"] = f"{original_title}_augmented_{i}"

            # Columns with sigma depending on sign
            for col in cols_sign_sigma:
                value = row[col]
                sigma = 0.00005 if value < 0 else 0.05
                new_row[col] = value + np.random.normal(0, sigma)

            # TD: fixed sigma
            new_row["TD"] = row["TD"] + np.random.normal(0, 0.05)

            # Frequency: fixed sigma
            new_row["Frequency"] = row["Frequency"] + np.random.normal(0, 0.01)

            final_rows.append(new_row)

    return pd.DataFrame(final_rows)
