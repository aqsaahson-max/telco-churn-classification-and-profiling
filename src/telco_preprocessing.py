
"""Shared preprocessing definitions for Group 2026-DS-77."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

PREDICTOR_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_predictors(frame):
    """Validate predictor columns and apply fixed numeric parsing.

    This function learns no statistics and never modifies its input.
    Missing TotalCharges values remain missing for later imputation.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame of predictor columns.")

    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names found.")

    missing = set(PREDICTOR_COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(PREDICTOR_COLUMNS)

    if missing or unexpected:
        raise ValueError(
            f"Predictor schema mismatch. "
            f"Missing: {sorted(missing)}; "
            f"Unexpected: {sorted(unexpected)}"
        )

    # Select a consistent column order and copy before making changes.
    result = frame.loc[:, PREDICTOR_COLUMNS].copy()

    text = result["TotalCharges"].astype("string").str.strip()
    blank = text.isna() | text.eq("").fillna(False)

    numeric = pd.to_numeric(text.mask(blank), errors="coerce")
    unexpected_text = numeric.isna() & ~blank

    if unexpected_text.any():
        raise ValueError("Unexpected non-numeric TotalCharges values found.")

    result["TotalCharges"] = numeric.astype("float64")

    # Our current categorical policy assumes complete values.
    # Stop for review if a later input violates that assumption.
    categories = result[CATEGORICAL_FEATURES]
    if categories.isna().any().any():
        raise ValueError("Missing categorical values require review.")

    for column in CATEGORICAL_FEATURES:
        if result[column].astype("string").str.strip().eq("").any():
            raise ValueError(f"Blank categorical value in {column}.")

    return result


def make_model_preprocessor(scale_numeric=True):
    """Return a NEW, UNFITTED pipeline for the 19 raw predictors."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    columns = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=None,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline([
        (
            "parse",
            FunctionTransformer(parse_predictors, validate=False),
        ),
        ("columns", columns),
    ])
