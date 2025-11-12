# src/data/load_data.py
import pandas as pd
from pathlib import Path

def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the credit risk dataset from a given path and perform basic validation.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)
    df = df.drop_duplicates()

    if 'loan_status' not in df.columns:
        raise ValueError("Target column 'loan_status' is missing from the dataset.")

    print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
