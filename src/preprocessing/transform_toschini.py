import numpy as np

def transform_toschini(df, crypt_threshold=50):
    """
    Transform the TOSCHINI dataset by extracting relevant columns, computing
    derived features, and formatting them to match the unified dataset structure.

    Args:
        df (pandas.DataFrame): Raw TOSCHINI dataset loaded from Excel.

    Returns:
        dict: A dictionary containing processed numpy arrays for each feature
              required in the final merged dataset.
    """

    arrays = {col: df[col].to_numpy() for col in df.columns}

    paper = arrays["paper"]
    ADR = arrays["ADR [Gy/s]"]
    macropulsewidth = arrays["Macropulse width [s]"]
    DPP = arrays["DPP [Gy]"]
    macropulsefrequency = arrays["Macropulse frequency [Hz]"]
    maxdose = arrays["max Dose [Gy]"]
    mindose = arrays["min Dose [Gy]"]
    deliverytime = arrays["delivery time [s]"]
    particle = arrays["particle"]
    crypt = arrays["%crypt"]

    new_crypt = np.where(crypt < crypt_threshold, 1, 0)
    target = np.full(len(particle), "Mice intestine")

    TD = (maxdose + mindose) / 2

    return {
        "Title": paper,
        "MDR": ADR,
        "PW": macropulsewidth,
        "DPP": DPP,
        "Frequency": macropulsefrequency,
        "NoP": arrays["noP"],
        "TD": TD,
        "Time": deliverytime,
        "Particle": particle,
        "Target": target,
        "Endpoint": new_crypt
    }
