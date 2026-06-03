import pandas as pd

def load_excel(path):
    """
    Load an Excel file and return its contents as a pandas DataFrame.

    Args:
        path (str): Path to the Excel file to be loaded.

    Returns:
        pandas.DataFrame: The data contained in the Excel file.
    """
    return pd.read_excel(path)


def save_excel(df, path):
    """
    Save a pandas DataFrame to an Excel file.

    Args:
        df (pandas.DataFrame): DataFrame to save.
        path (str): Output file path.
    """
    df.to_excel(path, index=False)
