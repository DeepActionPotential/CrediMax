# src/data/split.py
from sklearn.model_selection import train_test_split

def split_data(df, target_col: str = "loan_status", test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into train/test sets using stratification.
    Returns X_train, X_test, y_train, y_test.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f" Split done: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")
    return X_train, X_test, y_train, y_test
