import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.feature_engineering import feature_engineering_pipeline


def remove_duplicates(data):
    duplicate_count = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        data = data.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicate_count} duplicate rows.")
    else:
        print("No duplicate rows found.")
    return data


def build_preprocessor():
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Data split completed.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def preprocess_data(data):
    data = remove_duplicates(data)
    data = feature_engineering_pipeline(data)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"Preprocessor saved to {PREPROCESSOR_FILE}")

    feature_names = (
        NUMERICAL_FEATURES
        + list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    return X_train_df, X_test_df, y_train, y_test, preprocessor
