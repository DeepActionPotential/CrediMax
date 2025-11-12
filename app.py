from src.data.load_data import load_dataset
from src.data.split import split_data
from src.data.preprocess import feature_engineering, build_preprocessor

# Load dataset
df = load_dataset("data/raw/credit_risk_dataset.csv")

# Feature engineering
df_fe = feature_engineering(df)

# Split
X_train, X_test, y_train, y_test = split_data(df_fe, target_col="loan_status")

# Build preprocessing pipeline
preprocessor = build_preprocessor(X_train)
