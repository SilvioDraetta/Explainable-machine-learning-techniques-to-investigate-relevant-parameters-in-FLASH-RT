import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_excel, save_excel
import pandas as pd
import numpy as np
from src.preprocessing.encode_features import encode_target
from src.preprocessing.scale_data import scale_dataset
from src.preprocessing.augment_data import augment_dataset

df = load_excel("data/raw/electrons-protons/PREZADO-MICE-FLASH-INVIVO-NORMAL.xlsx")

def transform_prezado_only(df):
    """
    Transform the PREZADO dataset by extracting relevant columns and computing
    derived features needed for using only the Prezado dataset.

    Args:
        df (pandas.DataFrame): (selected) Raw PREZADO dataset loaded from Excel.

    Returns:
        dict: A dictionary containing processed numpy arrays for each feature
              required in the final merged dataset.
    """

    arrays = {col: df[col].to_numpy() for col in df.columns}

    Title = arrays["Title"]
    MDR = arrays["Mean Dose Rate (Gy/s)"]
    PW = arrays["Pulse Width (μs)"]
    PD = arrays["Pulse Dose (Gy)"]
    Repetition = arrays["Repetition Frequency (Hz)"]
    Pulses = arrays["Number of Pulses"]
    TD = arrays["Total Dose (Gy)"]
    Duration = arrays["Total Duration (s)"]
    Rad_type = arrays["Rad. Type"]
    Target = encode_target(arrays["Biological Target"])
    NTSS = arrays["NTSS"]

    new_Rad_type = np.where(Rad_type == "Electrons", 1, 0)

    return {
        "Title": Title,
        "MDR": MDR,
        "PW": PW,
        "DPP": PD,
        "Frequency": Repetition,
        "NoP": Pulses,
        "TD": TD,
        "Time": Duration,
        "Particle": new_Rad_type,
        "Target": Target,
        "Endpoint": NTSS
    }

transformed_data = transform_prezado_only(df)

# Convert dict to DataFrame
transformed_df = pd.DataFrame(transformed_data)

transformed_df = transformed_df.replace("-", np.nan)

# Scale
scaled_data = scale_dataset(transformed_df)

# Augment
augmented_data = augment_dataset(scaled_data, num_variants=4)

# Save
save_excel(augmented_data, "data/processed/processed_elec-prot/prezado_only.xlsx")
