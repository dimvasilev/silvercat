from dataclasses import dataclass
from pathlib import Path
import shutil

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    processed_dir: Path
    models_dir: Path
    accepts: Path
    rejects: Path
    data_dict: Path


def setup(
    project_root: str | Path | None = None,
    download_data: bool = True,
) -> ProjectPaths:
    """Set up project paths and optionally download the Lending Club dataset."""

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    models_dir = data_dir / "models"

    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        accepts=data_dir / "accepted_2007_to_2018Q4.csv.gz",
        rejects=data_dir / "rejected_2007_to_2018Q4.csv.gz",
        data_dict=project_root / "LCDataDictionary.csv",
    )

    if download_data and not paths.accepts.exists():
        kaggle_path = Path(
            kagglehub.dataset_download("wordsforthewise/lending-club")
        )

        for file in kaggle_path.iterdir():
            if file.is_file():
                shutil.copy2(file, data_dir / file.name)

        print(f"Successfully copied Lending Club files to {data_dir}")
    elif paths.accepts.exists():
        print("Lending Club data already exists")

    print("Project root:", paths.project_root)
    print("Data directory:", paths.data_dir)
    print("Accepted data exists:", paths.accepts.exists())
    print("Rejected data exists:", paths.rejects.exists())
    print("Data dictionary exists:", paths.data_dict.exists())

    return paths


def add_loan_date_features(
    df: pd.DataFrame,
    issue_col: str = "issue_d",
    last_payment_col: str = "last_pymnt_d",
) -> pd.DataFrame:
    """Add common date features used across the analytics notebooks."""

    df = df.copy()

    df["issue_month"] = pd.to_datetime(
        df[issue_col],
        format="%b-%Y",
        errors="coerce",
    )

    df["issue_year"] = df["issue_month"].dt.year.astype("Int16")

    if last_payment_col in df.columns:
        df["last_pymnt_month"] = pd.to_datetime(
            df[last_payment_col],
            format="%b-%Y",
            errors="coerce",
        )

        df["months_on_book"] = (
            (df["last_pymnt_month"].dt.year - df["issue_month"].dt.year) * 12
            + (df["last_pymnt_month"].dt.month - df["issue_month"].dt.month)
        )

    return df


def object_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missingness, cardinality and top values for object columns."""

    object_cols = df.select_dtypes(include=["object", "string"]).columns

    summary = pd.DataFrame({
        "missing_pct": df[object_cols].isna().mean().mul(100).round(1),
        "unique_values": df[object_cols].nunique(dropna=True),
        "top_value": df[object_cols].mode(dropna=True).iloc[0],
        "top_value_pct": (
            df[object_cols]
            .apply(lambda x: x.value_counts(dropna=True).iloc[0] / x.notna().sum() * 100)
            .round(1)
        ),
    })

    return summary.sort_values("unique_values", ascending=False)


def missingness_by_year(
    df: pd.DataFrame,
    year_col: str = "issue_year",
) -> pd.DataFrame:
    """Return percentage missing by variable and issue year."""

    missing = (
        df.groupby(year_col)
        .apply(lambda x: x.isna().mean().mul(100))
        .T
        .round(1)
    )

    return missing.drop(index=year_col, errors="ignore")


def distribution_by_year(
    df: pd.DataFrame,
    segment_col: str,
    year_col: str = "issue_year",
    percent: bool = True,
) -> pd.DataFrame:
    """Return counts or percentages for a segment by year."""

    counts = (
        df
        .groupby([year_col, segment_col], observed=True)
        .size()
        .unstack(year_col, fill_value=0)
    )

    if percent:
        return (
            counts
            .div(counts.sum(axis=0), axis=1)
            .mul(100)
            .round(1)
        )

    return counts.astype(int)


def apply_terminal_target(
    df: pd.DataFrame,
    status_col: str = "loan_status",
    target_col: str = "bad",
) -> pd.DataFrame:
    """Create a terminal-status bad flag and remove unresolved outcomes."""

    bad_statuses = ["Charged Off", "Default"]
    good_statuses = ["Fully Paid"]

    df = df.copy()

    df["target_status"] = np.select(
        [
            df[status_col].isin(bad_statuses),
            df[status_col].isin(good_statuses),
        ],
        ["bad", "good"],
        default="exclude",
    )

    return (
        df
        .loc[lambda d: d["target_status"].isin(["good", "bad"])]
        .assign(**{target_col: lambda d: d["target_status"].eq("bad").astype(int)})
        .copy()
    )


def bad_rate_summary(
    df: pd.DataFrame,
    year_col: str = "issue_year",
    target_col: str = "bad",
) -> pd.DataFrame:
    """Summarise count, bad count and bad rate by year."""

    summary = (
        df
        .groupby(year_col)
        .agg(
            loans=(target_col, "size"),
            bads=(target_col, "sum"),
            bad_rate=(target_col, "mean"),
        )
    )

    summary["bad_rate"] = summary["bad_rate"].mul(100).round(1)

    return summary


def bin_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add segment bands used for target diagnostics."""

    df = df.copy()

    df["loan_amount_band"] = pd.qcut(
        df["loan_amnt"],
        q=4,
        duplicates="drop",
    )

    df["int_rate_band"] = pd.qcut(
        df["int_rate"],
        q=4,
        duplicates="drop",
    )

    df["fico_mid"] = (
        df["fico_range_low"] + df["fico_range_high"]
    ) / 2

    df["fico_band"] = pd.cut(
        df["fico_mid"],
        bins=[0, 660, 700, 740, 780, 850],
        labels=["<660", "660-699", "700-739", "740-779", "780+"],
        right=False,
    )

    return df


def plot_bad_rate_by_year(
    df: pd.DataFrame,
    segment_cols: list[str],
    target_col: str = "bad",
    year_col: str = "issue_year",
) -> None:
    """Plot overall and segment-level bad rates by origination year."""

    plot_specs = [None] + segment_cols

    for segment_col in plot_specs:
        group_cols = [year_col] if segment_col is None else [year_col, segment_col]

        plot_df = (
            df.groupby(group_cols, observed=True)[target_col]
            .mean()
            .mul(100)
            .reset_index(name="bad_rate")
        )

        plt.figure(figsize=(14, 7))

        sns.lineplot(
            data=plot_df,
            x=year_col,
            y="bad_rate",
            hue=segment_col,
            marker="o",
        )

        title = (
            "Overall Bad Rate Over Issue Year"
            if segment_col is None
            else f"Bad Rate by {segment_col} Over Issue Year"
        )

        plt.title(title)
        plt.xlabel("Issue Year")
        plt.ylabel("Bad Rate (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        if segment_col is not None:
            plt.legend(title=segment_col, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()


def add_model_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple feature transformations used for univariate model-candidate review."""

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

    df["fico_avg"] = (
        df["fico_range_low"] + df["fico_range_high"]
    ) / 2

    return df.drop(
        columns=["earliest_cr_line", "fico_range_low", "fico_range_high", "issue_month"],
        errors="ignore",
    )


def plot_separation_grid(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str = "bad",
    bins: int = 5,
    min_count: int = 100,
    ncols: int = 4,
) -> None:
    """Plot volume and bad-rate separation charts for candidate variables."""

    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
    )

    axes = np.array(axes).reshape(-1)

    for ax1, col in zip(axes, cols):
        x = df[[col, target_col]].copy()

        if pd.api.types.is_numeric_dtype(x[col]):
            x["bin"] = pd.NA

            not_missing = x[col].notna()

            x.loc[not_missing, "bin"] = pd.qcut(
                x.loc[not_missing, col],
                q=bins,
                duplicates="drop",
            ).astype(str)

            x.loc[~not_missing, "bin"] = "Missing"

        else:
            x["bin"] = x[col].fillna("Missing").astype(str)

        summary = (
            x.groupby("bin", observed=True)
            .agg(
                volume=(target_col, "size"),
                bad_rate=(target_col, "mean"),
            )
            .reset_index()
            .query("volume >= @min_count")
            .assign(bad_rate=lambda d: d["bad_rate"].mul(100))
        )

        sns.barplot(
            data=summary,
            x="bin",
            y="volume",
            ax=ax1,
            alpha=0.35,
        )

        ax2 = ax1.twinx()

        sns.lineplot(
            data=summary,
            x="bin",
            y="bad_rate",
            marker="o",
            ax=ax2,
        )

        ax1.set_title(col)
        ax1.set_xlabel("")
        ax1.set_ylabel("Volume")
        ax2.set_ylabel("Bad rate (%)")
        ax1.tick_params(axis="x", rotation=45, labelsize=8)

    for ax in axes[len(cols):]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def pct_mix(
    df: pd.DataFrame,
    index_col: str,
    segment_col: str,
) -> pd.DataFrame:
    """Return percentage mix of a segment by index column."""

    return (
        pd.crosstab(df[index_col], df[segment_col], normalize="index")
        .mul(100)
        .round(1)
    )


def yearly_summary(
    df: pd.DataFrame,
    cols: list[str],
    year_col: str = "issue_year",
) -> pd.DataFrame:
    """Return count, mean and median for selected columns by year."""

    return (
        df.groupby(year_col)[cols]
        .agg(["count", "mean", "median"])
        .round(2)
    )


def plot_mix(
    df: pd.DataFrame,
    segment_col: str,
    title: str,
    year_col: str = "issue_year",
) -> None:
    """Plot stacked portfolio mix by year."""

    mix = pct_mix(df, year_col, segment_col)

    ax = mix.plot(kind="bar", stacked=True, figsize=(14, 6))
    ax.set_title(title)
    ax.set_xlabel("Issue year")
    ax.set_ylabel("Portfolio mix (%)")
    plt.legend(title=segment_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_yearly_metric(
    df: pd.DataFrame,
    col: str,
    title: str,
    year_col: str = "issue_year",
    agg: str = "median",
) -> None:
    """Plot a yearly aggregated metric."""

    summary = (
        df.groupby(year_col)[col]
        .agg(agg)
        .reset_index()
    )

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=summary, x=year_col, y=col, marker="o")
    plt.title(title)
    plt.xlabel("Issue year")
    plt.ylabel(col)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_interest_rate_by_grade(
    df: pd.DataFrame,
    year_col: str = "issue_year",
    grade_col: str = "grade",
    rate_col: str = "int_rate",
) -> None:
    """Plot median interest rate by grade and issue year."""

    int_rate_by_grade = (
        df
        .groupby([year_col, grade_col], observed=True)[rate_col]
        .median()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=int_rate_by_grade,
        x=year_col,
        y=rate_col,
        hue=grade_col,
        marker="o",
    )

    plt.title("Median Interest Rate by Grade and Issue Year")
    plt.xlabel("Issue year")
    plt.ylabel("Median interest rate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def count_applications_by_year(
    file_path: str | Path,
    date_col: str,
    output_col: str,
    compression: str = "gzip",
    date_format: str | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Count applications by year from a large CSV using chunks."""

    chunks = []

    for chunk in pd.read_csv(
        file_path,
        compression=compression,
        usecols=[date_col],
        chunksize=chunksize,
    ):
        chunk["year"] = pd.to_datetime(
            chunk[date_col],
            format=date_format,
            errors="coerce",
        ).dt.year

        chunks.append(chunk.groupby("year").size())

    return (
        pd.concat(chunks, axis=1)
        .sum(axis=1)
        .reset_index(name=output_col)
    )


def build_application_volume_summary(
    accepted_volume: pd.DataFrame,
    rejected_volume: pd.DataFrame,
) -> pd.DataFrame:
    """Combine accepted and rejected yearly volumes and calculate proxy acceptance rate."""

    application_volume = (
        accepted_volume
        .merge(rejected_volume, on="year", how="outer")
        .fillna(0)
        .sort_values("year")
    )

    application_volume["accepted"] = application_volume["accepted"].astype(int)
    application_volume["rejected"] = application_volume["rejected"].astype(int)
    application_volume["total_applications"] = (
        application_volume["accepted"] + application_volume["rejected"]
    )

    application_volume["acceptance_rate"] = (
        application_volume["accepted"]
        / application_volume["total_applications"]
        * 100
    ).round(1)

    application_volume["year"] = application_volume["year"].astype(int)

    return application_volume.sort_values("year")


def millions_formatter(x: float, pos: int) -> str:
    """Format large chart axes as k / m."""

    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}m".replace(".0m", "m")
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


def plot_application_volume_and_acceptance(
    application_volume: pd.DataFrame,
) -> None:
    """Plot total application volume and proxy acceptance rate by year."""

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.bar(
        application_volume["year"],
        application_volume["total_applications"],
        alpha=0.35,
    )

    ax2 = ax1.twinx()

    ax2.plot(
        application_volume["year"],
        application_volume["acceptance_rate"],
        marker="o",
    )

    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    ax1.set_title("Total Application Volume and Proxy Acceptance Rate by Year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total application volume")
    ax2.set_ylabel("Acceptance rate (%)")

    ax1.set_xticks(application_volume["year"])
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
