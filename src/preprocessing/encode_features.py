import pandas as pd

mapping = {
    "Mice brain": 1,
    "Mice ear": 6,
    "Mice intestine": 0,
    "Mice pelvis": 7,
    "Mice heart": 8,
    "Mice whole abdomen": 3,
    "Mice whole thorax": 4,
    "Rat feet": 2,
    "Mouse foot skin": 2,
    "Mice leg": 5,
    "Rat brain": 1
}

def encode_target(arr):
    """
    Encode categorical biological target labels into numerical values using
    a predefined mapping.

    Args:
        arr (array-like): Array of target labels as strings.

    Returns:
        numpy.ndarray: Array of encoded numerical target values.
    """

    return pd.Series(arr).map(mapping).to_numpy()
