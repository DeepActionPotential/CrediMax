# src/data/preprocess.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform lightweight feature engineering: logs and ratios.
    """
    df = df.copy()

    if 'person_income' in df.columns:
        df['log_person_income'] = np.log1p(df['person_income'].clip(lower=0))
    if 'loan_amnt' in df.columns:
        df['log_loan_amnt'] = np.log1p(df['loan_amnt'].clip(lower=0))
    if {'loan_amnt', 'person_income'}.issubset(df.columns):
        df['loan_to_income'] = df['loan_amnt'] / df['person_income'].replace(0, np.nan)

    return df


def build_preprocessor(X: pd.DataFrame):
    """
    Build a sklearn ColumnTransformer with numeric and categorical preprocessing.
    """
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    print(f" Preprocessor built with {len(num_cols)} numeric and {len(cat_cols)} categorical features")
    return preprocessor
