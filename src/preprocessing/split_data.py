import pandas as pd

def split_dataset(df, prezado_count=53, val_fraction=0.25, seed1=1, seed2=2):
    """
    Split the dataset into training and validation sets by sampling a fixed
    percentage from the PREZADO block and the TOSCHINI block separately.

    Args:
        df (pandas.DataFrame): The full dataset.
        prezado_count (int): Number of rows belonging to the PREZADO dataset.
        val_fraction (float): Fraction of each block to include in validation.
        seed1 (int): Random seed for PREZADO sampling.
        seed2 (int): Random seed for TOSCHINI sampling.

    Returns:
        tuple: (df_train, df_val) where both are pandas DataFrames.
    """
    # Split blocks
    prezado_block = df.iloc[:prezado_count]
    toschini_block = df.iloc[prezado_count:]

    # Sample validation subsets
    val_prezado = prezado_block.sample(frac=val_fraction, random_state=seed1)
    val_toschini = toschini_block.sample(frac=val_fraction, random_state=seed2)

    # Combine validation
    df_val = pd.concat([val_prezado, val_toschini], axis=0)

    # Training = everything else
    df_train = df.drop(df_val.index)

    return df_train, df_val
