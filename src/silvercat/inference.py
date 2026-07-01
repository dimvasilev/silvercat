from pathlib import Path
import joblib
import pandas as pd


def load_model(model_path="models/credit_risk_model.joblib"):
    """Load the fitted model and preprocessing metadata."""
    return joblib.load(model_path)


def _to_dataframe(application_data):
    """Convert one record, a list of records, or a DataFrame into a DataFrame."""
    if isinstance(application_data, pd.DataFrame):
        return application_data.copy()

    if isinstance(application_data, dict):
        return pd.DataFrame([application_data])

    return pd.DataFrame(application_data)


def preprocess_applications(application_data, artifacts):
    """Apply the same feature engineering used during model development."""
    df = _to_dataframe(application_data)

    candidate_vars = artifacts["candidate_vars"]
    categorical_cols = artifacts["categorical_cols"]
    sparse_adverse_vars = artifacts["sparse_adverse_vars"]
    segment_cols = artifacts["segment_cols"]
    drop_cols = artifacts["drop_cols"]
    feature_cols = artifacts["feature_cols"]

    missing_candidate_cols = [col for col in candidate_vars if col not in df.columns]
    if missing_candidate_cols:
        raise ValueError(
            "Input data is missing required model fields: "
            + ", ".join(missing_candidate_cols)
        )

    if "issue_month" in df.columns:
        df["issue_month"] = pd.to_datetime(df["issue_month"], errors="coerce")
    elif "issue_d" in df.columns:
        df["issue_month"] = pd.to_datetime(
            df["issue_d"],
            format="%b-%Y",
            errors="coerce",
        )
    else:
        raise ValueError("Input data must include either issue_d or issue_month.")

    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )

    df["account_age_mths"] = (
        (df["issue_month"].dt.year - df["earliest_cr_line"].dt.year) * 12
        + (df["issue_month"].dt.month - df["earliest_cr_line"].dt.month)
    )

    df["fico_avg"] = (
        df["fico_range_low"] + df["fico_range_high"]
    ) / 2

    for col in sparse_adverse_vars:
        df[f"{col}_flag"] = (
            df[col]
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Missing"

    df[categorical_cols] = df[categorical_cols].fillna("Missing")

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
    )

    df = df.drop(
        columns=[*drop_cols, "bad", "target_status", "issue_month", *segment_cols],
        errors="ignore",
    )

    X = df.reindex(columns=feature_cols, fill_value=0)

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X


def predict(application_data, artifacts=None, model_path="models/credit_risk_model.joblib"):
    """Return predicted bad probability for application data."""
    if artifacts is None:
        artifacts = load_model(model_path)

    X = preprocess_applications(application_data, artifacts)

    pbad = artifacts["model"].predict_proba(X)[:, 1]

    return pd.DataFrame({
        "pbad": pbad,
    })
