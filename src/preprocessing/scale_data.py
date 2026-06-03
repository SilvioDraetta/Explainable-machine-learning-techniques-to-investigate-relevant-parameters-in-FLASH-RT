import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_dataset(df):
    """
    Apply standard scaling to numerical columns of the dataset.

    Args:
        df (pandas.DataFrame): Input dataset containing numerical features.

    Returns:
        pandas.DataFrame: Scaled dataset with the same structure as the input.
    """
    scaler = StandardScaler()

    numeric_cols = ["MDR", "PW", "DPP", "Frequency", "NoP", "TD", "Time", "Target"]

    scaled = df.copy()

    for col in numeric_cols:
        scaled[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1)).flatten()

    return scaled
