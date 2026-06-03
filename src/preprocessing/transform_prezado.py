import numpy as np

def transform_prezado(df):
    """
    Transform the PREZADO dataset by extracting relevant columns and computing
    derived features needed for the unified dataset.

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
    Target = arrays["Biological Target"]
    NTSS = arrays["NTSS"]

    new_Rad_type = np.where(Rad_type == "Electrons", 1, 0)
    new_NTSS = np.where(NTSS < 3.5, 1, 0)

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
        "Endpoint": new_NTSS
    }
