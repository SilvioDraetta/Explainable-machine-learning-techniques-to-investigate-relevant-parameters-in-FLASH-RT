import numpy as np
import pandas as pd
from .encode_features import encode_target

def merge_datasets(prezado, toschini):
    """
    Merge the processed PREZADO and TOSCHINI datasets into a single pandas
    DataFrame, aligning all features and applying target encoding.

    Args:
        prezado (dict): Processed PREZADO data returned by transform_prezado().
        toschini (dict): Processed TOSCHINI data returned by transform_toschini().

    Returns:
        pandas.DataFrame: The final merged dataset ready for saving or modeling.
    """

    df = pd.DataFrame({
        "Title": np.concatenate((prezado["Title"], toschini["Title"])),
        "MDR": np.concatenate((prezado["MDR"], toschini["MDR"])),
        "PW": np.concatenate((prezado["PW"], toschini["PW"])),
        "DPP": np.concatenate((prezado["DPP"], toschini["DPP"])),
        "Frequency": np.concatenate((prezado["Frequency"], toschini["Frequency"])),
        "NoP": np.concatenate((prezado["NoP"], toschini["NoP"])),
        "TD": np.concatenate((prezado["TD"], toschini["TD"])),
        "Time": np.concatenate((prezado["Time"], toschini["Time"])),
        "Particle": np.concatenate((prezado["Particle"], toschini["Particle"])),
        "Target": encode_target(np.concatenate((prezado["Target"], toschini["Target"]))),
        "Endpoint": np.concatenate((prezado["Endpoint"], toschini["Endpoint"]))
    })

    df = df.replace("-", np.nan)
    return df
