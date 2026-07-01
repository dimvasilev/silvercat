from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna


CANDIDATE_VARS = [
    "home_ownership", "purpose",
    "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mths_since_last_major_derog", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_bc_dlq",
    "mths_since_recent_inq", "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "term", "loan_amnt", "int_rate",
]

SPARSE_ADVERSE_VARS = [
    "acc_now_delinq",
    "tot_coll_amt",
    "pub_rec",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "pub_rec_bankruptcies",
    "tax_liens",
]

SEGMENT_COLS = ["term", "loan_amnt", "int_rate"]

DROP_COLS = [
    "issue_d",
    "loan_status",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    *SPARSE_ADVERSE_VARS,
]


def load_accepted_modelling_data(
    accepts_path: str | Path,
    candidate_vars: list[str] = CANDIDATE_VARS,
) -> pd.DataFrame:
    """Read accepted applications required for model development."""

    usecols = list(dict.fromkeys([*candidate_vars, "issue_d", "loan_status"]))

    return pd.read_csv(
        accepts_path,
        compression="gzip",
        usecols=usecols,
        low_memory=False,
    )


def create_modelling_target(
    df: pd.DataFrame,
    start: str = "2014-01-01",
    end: str = "2016-06-30",
) -> pd.DataFrame:
    """Add issue month and bad flag, keeping only terminal good/bad outcomes."""

    bad_statuses = ["Charged Off", "Default"]
    good_statuses = ["Fully Paid"]

    df = df.copy()
    df["issue_month"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    return (
        df
        .loc[lambda d: d["issue_month"].between(start, end)]
        .loc[lambda d: d["loan_status"].isin(bad_statuses + good_statuses)]
        .assign(bad=lambda d: d["loan_status"].isin(bad_statuses).astype(int))
        .copy()
    )


def add_model_features(
    df: pd.DataFrame,
    sparse_adverse_vars: list[str] = SPARSE_ADVERSE_VARS,
) -> pd.DataFrame:
    """Create engineered model features used in development and inference."""

    df = df.copy()

    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )

    df["account_age_mths"] = (
        (df["issue_month"].dt.year - df["earliest_cr_line"].dt.year) * 12
        + (df["issue_month"].dt.month - df["earliest_cr_line"].dt.month)
    )

    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    for col in sparse_adverse_vars:
        df[f"{col}_flag"] = df[col].fillna(0).gt(0).astype(int)

    return df


def get_categorical_cols(
    df: pd.DataFrame,
    segment_cols: list[str] = SEGMENT_COLS,
) -> list[str]:
    """Identify categorical variables to one-hot encode."""

    excluded = {"issue_d", "loan_status", "target_status", *segment_cols}

    return [
        col for col in df.select_dtypes(include=["object", "string"]).columns
        if col not in excluded
    ]


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """One-hot encode categorical variables using pandas."""

    df = df.copy()
    df[categorical_cols] = df[categorical_cols].fillna("Missing")

    return pd.get_dummies(df, columns=categorical_cols, drop_first=False)


def prepare_modelling_frame(
    df: pd.DataFrame,
    sparse_adverse_vars: list[str] = SPARSE_ADVERSE_VARS,
    segment_cols: list[str] = SEGMENT_COLS,
    drop_cols: list[str] = DROP_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature engineering, encoding and source-field drops."""

    df = add_model_features(df, sparse_adverse_vars=sparse_adverse_vars)
    categorical_cols = get_categorical_cols(df, segment_cols=segment_cols)
    df = encode_categoricals(df, categorical_cols)

    return df.drop(columns=drop_cols, errors="ignore"), categorical_cols


def time_split(
    df: pd.DataFrame,
    date_col: str = "issue_month",
) -> dict[str, pd.DataFrame]:
    """Create train, test and out-of-time samples."""

    return {
        "train": df.loc[df[date_col].between("2014-01-01", "2015-06-30")].copy(),
        "test": df.loc[df[date_col].between("2015-07-01", "2015-12-31")].copy(),
        "oot": df.loc[df[date_col].between("2016-01-01", "2016-06-30")].copy(),
    }


def split_summary(splits: dict[str, pd.DataFrame], target_col: str = "bad") -> pd.DataFrame:
    """Summarise sample size and bad rate for each split."""

    return pd.DataFrame({
        "sample": list(splits),
        "rows": [len(df) for df in splits.values()],
        "bad_rate": [df[target_col].mean() * 100 for df in splits.values()],
    }).assign(bad_rate=lambda d: d["bad_rate"].round(1))


def make_feature_matrices(
    splits: dict[str, pd.DataFrame],
    exclude_cols: list[str] | None = None,
    target_col: str = "bad",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], list[str]]:
    """Return X/y dictionaries and the fitted feature column list."""

    if exclude_cols is None:
        exclude_cols = [target_col, "issue_month", *SEGMENT_COLS]

    feature_cols = [col for col in splits["train"].columns if col not in exclude_cols]

    X = {name: df[feature_cols].copy() for name, df in splits.items()}
    y = {name: df[target_col] for name, df in splits.items()}

    bool_cols = X["train"].select_dtypes(include="bool").columns

    for name in X:
        X[name][bool_cols] = X[name][bool_cols].astype(int)

    return X, y, feature_cols


def fit_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    params: dict | None = None,
    verbose: int | bool = 50,
) -> XGBClassifier:
    """Fit an XGBoost binary classifier."""

    default_params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(**(params or default_params))
    eval_set = None if X_valid is None else [(X_valid, y_valid)]

    model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

    return model


def score_splits(
    splits: dict[str, pd.DataFrame],
    X: dict[str, pd.DataFrame],
    model: XGBClassifier,
    score_col: str,
) -> dict[str, pd.DataFrame]:
    """Add predicted bad probabilities to each split."""

    scored = {name: df.copy() for name, df in splits.items()}

    for name in scored:
        scored[name][score_col] = model.predict_proba(X[name])[:, 1]

    return scored


def tune_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_size: int = 300_000,
    n_trials: int = 3,
    random_state: int = 42,
) -> optuna.study.Study:
    """Run a lightweight Optuna search for XGBoost parameters."""

    sample_size = min(sample_size, len(X_train))
    sample_idx = X_train.sample(n=sample_size, random_state=random_state).index

    X_tune = X_train.loc[sample_idx]
    y_tune = y_train.loc[sample_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }

        model = fit_xgb_model(X_tune, y_tune, X_valid, y_valid, params, verbose=False)
        return roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study


def xgb_params_from_study(study: optuna.study.Study, random_state: int = 42) -> dict:
    """Return final XGBoost parameters from an Optuna study."""

    return {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }


def auc_gini(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> pd.Series:
    """Calculate AUC and Gini."""

    if y_true.nunique() < 2:
        return pd.Series({"auc": np.nan, "gini": np.nan})

    auc = roc_auc_score(y_true, y_pred)

    return pd.Series({"auc": auc, "gini": 2 * auc - 1})


def performance_summary(
    y: dict[str, pd.Series],
    splits: dict[str, pd.DataFrame],
    score_cols: dict[str, str],
) -> pd.DataFrame:
    """Return AUC and Gini by sample and model."""

    rows = []

    for model_name, score_col in score_cols.items():
        for sample, df in splits.items():
            metrics = auc_gini(y[sample], df[score_col])
            rows.append({
                "model": model_name,
                "sample": sample,
                "auc": metrics["auc"],
                "gini": metrics["gini"],
            })

    return pd.DataFrame(rows).round({"auc": 3, "gini": 3})


def add_validation_bands(
    df: pd.DataFrame,
    amount_col: str = "loan_amnt",
    rate_col: str = "int_rate",
    q: int = 5,
) -> pd.DataFrame:
    """Create validation bands for amount and interest rate."""

    df = df.copy()

    df["loan_amnt_band"] = pd.qcut(df[amount_col], q=q, duplicates="drop")
    df["int_rate_band"] = pd.qcut(df[rate_col], q=q, duplicates="drop")

    return df


def gini_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    target_col: str = "bad",
    score_col: str = "pbad_tuned",
) -> pd.DataFrame:
    """Calculate Gini and bad rate by segment."""

    rows = []

    for segment, group in df.groupby(segment_col, observed=True):
        rows.append({
            segment_col: segment,
            "rows": len(group),
            "bads": group[target_col].sum(),
            "bad_rate": group[target_col].mean() * 100,
            "gini": auc_gini(group[target_col], group[score_col])["gini"],
        })

    return (
        pd.DataFrame(rows)
        .assign(
            bad_rate=lambda d: d["bad_rate"].round(1),
            gini=lambda d: d["gini"].round(3),
        )
    )


def shap_importance_table(
    shap_values,
    feature_names: list[str],
    threshold: float = 1e-6,
) -> pd.DataFrame:
    """Return mean absolute SHAP values above a threshold."""

    values = getattr(shap_values, "values", shap_values)

    return (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        })
        .loc[lambda d: d["mean_abs_shap"] > threshold]
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .assign(mean_abs_shap=lambda d: d["mean_abs_shap"].round(4))
    )


def save_model_artifacts(
    model: XGBClassifier,
    feature_cols: list[str],
    categorical_cols: list[str],
    model_path: str | Path = "models/credit_risk_model.joblib",
    candidate_vars: list[str] = CANDIDATE_VARS,
    sparse_adverse_vars: list[str] = SPARSE_ADVERSE_VARS,
    segment_cols: list[str] = SEGMENT_COLS,
    drop_cols: list[str] = DROP_COLS,
) -> Path:
    """Save fitted model and preprocessing metadata for deployment."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model": model,
        "candidate_vars": candidate_vars,
        "categorical_cols": categorical_cols,
        "sparse_adverse_vars": sparse_adverse_vars,
        "segment_cols": segment_cols,
        "drop_cols": drop_cols,
        "feature_cols": feature_cols,
    }

    joblib.dump(artifacts, model_path, compress=3)

    return model_path


INFERENCE_HELPER_CODE = 'from pathlib import Path\nimport joblib\nimport pandas as pd\n\n\ndef load_model(model_path="models/credit_risk_model.joblib"):\n    """Load the fitted model and preprocessing metadata."""\n    return joblib.load(model_path)\n\n\ndef _to_dataframe(application_data):\n    """Convert one record, a list of records, or a DataFrame into a DataFrame."""\n    if isinstance(application_data, pd.DataFrame):\n        return application_data.copy()\n\n    if isinstance(application_data, dict):\n        return pd.DataFrame([application_data])\n\n    return pd.DataFrame(application_data)\n\n\ndef preprocess_applications(application_data, artifacts):\n    """Apply the same feature engineering used during model development."""\n    df = _to_dataframe(application_data)\n\n    candidate_vars = artifacts["candidate_vars"]\n    categorical_cols = artifacts["categorical_cols"]\n    sparse_adverse_vars = artifacts["sparse_adverse_vars"]\n    segment_cols = artifacts["segment_cols"]\n    drop_cols = artifacts["drop_cols"]\n    feature_cols = artifacts["feature_cols"]\n\n    missing_candidate_cols = [col for col in candidate_vars if col not in df.columns]\n    if missing_candidate_cols:\n        raise ValueError(\n            "Input data is missing required model fields: "\n            + ", ".join(missing_candidate_cols)\n        )\n\n    if "issue_month" in df.columns:\n        df["issue_month"] = pd.to_datetime(df["issue_month"], errors="coerce")\n    elif "issue_d" in df.columns:\n        df["issue_month"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")\n    else:\n        raise ValueError("Input data must include either issue_d or issue_month.")\n\n    df["earliest_cr_line"] = pd.to_datetime(\n        df["earliest_cr_line"],\n        format="%b-%Y",\n        errors="coerce",\n    )\n\n    df["account_age_mths"] = (\n        (df["issue_month"].dt.year - df["earliest_cr_line"].dt.year) * 12\n        + (df["issue_month"].dt.month - df["earliest_cr_line"].dt.month)\n    )\n\n    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2\n\n    for col in sparse_adverse_vars:\n        df[f"{col}_flag"] = df[col].fillna(0).gt(0).astype(int)\n\n    for col in categorical_cols:\n        if col not in df.columns:\n            df[col] = "Missing"\n\n    df[categorical_cols] = df[categorical_cols].fillna("Missing")\n\n    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)\n\n    df = df.drop(\n        columns=[*drop_cols, "bad", "target_status", "issue_month", *segment_cols],\n        errors="ignore",\n    )\n\n    X = df.reindex(columns=feature_cols, fill_value=0)\n\n    bool_cols = X.select_dtypes(include="bool").columns\n    X[bool_cols] = X[bool_cols].astype(int)\n\n    return X\n\n\ndef predict(application_data, artifacts=None, model_path="models/credit_risk_model.joblib"):\n    """Return predicted bad probability for application data."""\n    if artifacts is None:\n        artifacts = load_model(model_path)\n\n    X = preprocess_applications(application_data, artifacts)\n    pbad = artifacts["model"].predict_proba(X)[:, 1]\n\n    return pd.DataFrame({"pbad": pbad})\n'

def write_inference_helper(path: str | Path = "src/silvercat/inference.py") -> Path:
    """Write the local inference helper used by the deployment notebook."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(INFERENCE_HELPER_CODE)

    return path


def cumulative_bad_capture(
    df: pd.DataFrame,
    target_col: str = "bad",
    score_col: str = "pbad_tuned",
) -> pd.DataFrame:
    """Return cumulative bad capture curve data."""

    gains = (
        df
        .copy()
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )

    gains["application_count"] = np.arange(1, len(gains) + 1)
    gains["cum_app_share"] = gains["application_count"] / len(gains)
    gains["cum_bads"] = gains[target_col].cumsum()
    gains["cum_bad_capture"] = gains["cum_bads"] / gains[target_col].sum()
    gains["random_capture"] = gains["cum_app_share"]

    return gains


def plot_bad_capture_curve(gains_df: pd.DataFrame) -> None:
    """Plot cumulative bad capture against random selection."""

    plt.figure(figsize=(8, 6))

    plt.plot(
        gains_df["cum_app_share"] * 100,
        gains_df["cum_bad_capture"] * 100,
        label="Model",
    )

    plt.plot(
        gains_df["cum_app_share"] * 100,
        gains_df["random_capture"] * 100,
        linestyle="--",
        label="Random selection",
    )

    plt.title("OOT Cumulative Bad Capture Curve")
    plt.xlabel("Applications ranked from highest to lowest predicted risk (%)")
    plt.ylabel("Actual bads captured (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def capture_summary(
    gains_df: pd.DataFrame,
    capture_points: list[float] | None = None,
    target_col: str = "bad",
) -> pd.DataFrame:
    """Summarise bad capture and lift at selected high-risk cut-offs."""

    if capture_points is None:
        capture_points = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    rows = []

    for point in capture_points:
        selected = gains_df.loc[gains_df["cum_app_share"] <= point]
        bad_capture = selected[target_col].sum() / gains_df[target_col].sum()

        rows.append({
            "highest_risk_applications": f"{int(point * 100)}%",
            "bad_capture": bad_capture * 100,
            "random_capture": point * 100,
            "lift_vs_random": bad_capture / point,
            "observed_bad_rate": selected[target_col].mean() * 100,
        })

    return pd.DataFrame(rows).round({
        "bad_capture": 1,
        "random_capture": 1,
        "lift_vs_random": 2,
        "observed_bad_rate": 1,
    })


def cutoff_summary(
    df: pd.DataFrame,
    target_col: str = "bad",
    score_col: str = "pbad_tuned",
    q: int = 10,
) -> pd.DataFrame:
    """Summarise bad rate and cumulative acceptance by predicted-risk band."""

    work = df.copy()
    work["risk_band"] = pd.qcut(work[score_col], q=q, labels=False, duplicates="drop") + 1

    summary = (
        work
        .groupby("risk_band")
        .agg(
            rows=(target_col, "size"),
            bads=(target_col, "sum"),
            min_pbad=(score_col, "min"),
            max_pbad=(score_col, "max"),
            avg_pbad=(score_col, "mean"),
            bad_rate=(target_col, "mean"),
        )
        .reset_index()
        .sort_values("risk_band")
    )

    summary["cum_rows"] = summary["rows"].cumsum()
    summary["cum_bads"] = summary["bads"].cumsum()
    summary["accept_rate"] = summary["cum_rows"] / summary["rows"].sum()
    summary["cum_bad_rate"] = summary["cum_bads"] / summary["cum_rows"]

    pct_cols = ["min_pbad", "max_pbad", "avg_pbad", "bad_rate", "accept_rate", "cum_bad_rate"]
    summary[pct_cols] = summary[pct_cols].mul(100).round(1)

    return summary


def plot_cutoff_tradeoff(summary: pd.DataFrame) -> None:
    """Plot acceptance rate against cumulative bad rate."""

    plt.figure(figsize=(8, 5))

    sns.lineplot(data=summary, x="accept_rate", y="cum_bad_rate", marker="o")

    plt.title("OOT Cut-off Simulation: Acceptance Rate vs Bad Rate")
    plt.xlabel("Cumulative acceptance rate (%)")
    plt.ylabel("Cumulative bad rate (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
