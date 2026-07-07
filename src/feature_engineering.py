"""
feature_engineering.py
-----------------------
Creates domain-specific features from the cleaned Ames Housing dataset.

All transformations are deterministic (no fitting required), so the same
function can be applied to both train and test DataFrames safely.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Helper: safe column getter (returns 0 if column is absent)
# ─────────────────────────────────────────────────────────────

def _get(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if present, else a zero Series of same length."""
    if col in df.columns:
        return df[col].fillna(0)
    return pd.Series(0, index=df.index)


# ─────────────────────────────────────────────────────────────
# Feature Engineering Functions
# ─────────────────────────────────────────────────────────────

def add_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate area-related features."""
    df["TotalSF"] = (
        _get(df, "TotalBsmtSF") +
        _get(df, "1stFlrSF")    +
        _get(df, "2ndFlrSF")
    )
    df["TotalPorchSF"] = (
        _get(df, "OpenPorchSF")    +
        _get(df, "EnclosedPorch")  +
        _get(df, "3SsnPorch")      +
        _get(df, "ScreenPorch")
    )
    df["TotalLivingArea"] = (
        _get(df, "GrLivArea") +
        _get(df, "TotalBsmtSF")
    )
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """House age and remodel age at year of sale."""
    yr_sold  = _get(df, "YrSold")
    yr_built = _get(df, "YearBuilt")
    yr_remod = _get(df, "YearRemodAdd")

    df["HouseAge"]  = (yr_sold - yr_built).clip(lower=0)
    df["RemodAge"]  = (yr_sold - yr_remod).clip(lower=0)
    df["IsRemodeled"] = ((yr_remod - yr_built) > 0).astype(int)
    df["IsNew"]       = (yr_sold == yr_built).astype(int)
    return df


def add_boolean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for presence of amenities."""
    df["HasBasement"]  = (_get(df, "TotalBsmtSF")  > 0).astype(int)
    df["HasGarage"]    = (_get(df, "GarageArea")    > 0).astype(int)
    df["HasPool"]      = (_get(df, "PoolArea")      > 0).astype(int)
    df["HasFireplace"] = (_get(df, "Fireplaces")    > 0).astype(int)
    df["HasDeck"]      = (
        (_get(df, "WoodDeckSF") + _get(df, "OpenPorchSF")) > 0
    ).astype(int)
    return df


def add_bathroom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted total bathroom count."""
    df["TotalBath"] = (
        _get(df, "FullBath")      +
        0.5 * _get(df, "HalfBath") +
        _get(df, "BsmtFullBath")  +
        0.5 * _get(df, "BsmtHalfBath")
    )
    return df


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features between quality scores and size."""
    overall_qual = _get(df, "OverallQual")
    overall_cond = _get(df, "OverallCond")
    gr_liv_area  = _get(df, "GrLivArea")

    df["TotalQual"]      = overall_qual * overall_cond
    df["GrLivArea_Qual"] = gr_liv_area  * overall_qual
    df["QualCondRatio"]  = overall_qual / (overall_cond + 1)
    df["PricePerSF"]     = overall_qual / (df.get("TotalSF", 1).replace(0, 1))
    return df


def add_year_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bin construction year into era categories."""
    yr_built = _get(df, "YearBuilt")
    bins   = [0, 1900, 1950, 1970, 1990, 2000, 2010, 2030]
    labels = [0,    1,    2,    3,    4,    5,    6]
    df["YearBuilt_Era"] = pd.cut(
        yr_built, bins=bins, labels=labels, right=True
    ).astype(float).fillna(0).astype(int)
    return df


def add_neighborhood_quality(df: pd.DataFrame,
                              neighborhood_rank: dict = None) -> pd.DataFrame:
    """
    Map neighborhood to a pre-computed rank (based on training mean SalePrice).
    If neighborhood_rank is None (e.g., during inference on test data where
    SalePrice is unknown), the mapping silently fills with 0.
    """
    if neighborhood_rank and "Neighborhood" in df.columns:
        df["Neighborhood_Quality"] = (
            df["Neighborhood"].map(neighborhood_rank).fillna(0)
        )
    else:
        df["Neighborhood_Quality"] = 0
    return df


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Id column if present."""
    return df.drop(columns=["Id"], errors="ignore")


# ─────────────────────────────────────────────────────────────
# Master Function
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      neighborhood_rank: dict = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Parameters
    ----------
    df                : Cleaned + encoded DataFrame.
    neighborhood_rank : dict mapping Neighborhood → rank (from training data).

    Returns
    -------
    DataFrame with additional engineered columns.
    """
    df = df.copy()
    df = add_area_features(df)
    df = add_age_features(df)
    df = add_boolean_features(df)
    df = add_bathroom_features(df)
    df = add_quality_features(df)
    df = add_year_category(df)
    df = add_neighborhood_quality(df, neighborhood_rank)
    df = drop_id(df)

    # Drop log-transformed target if accidentally present
    df.drop(columns=["SalePrice_log"], errors="ignore", inplace=True)

    # Convert any remaining object columns to int (edge cases)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Convert categorical dtype to int
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(float).astype(int)

    print(f"[✓] Feature engineering done. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# Build neighborhood rank map from training data
# ─────────────────────────────────────────────────────────────

def build_neighborhood_rank(train_df: pd.DataFrame,
                             target_col: str = "SalePrice") -> dict:
    """
    Compute neighborhood → rank dict from raw training DataFrame.
    """
    if "Neighborhood" not in train_df.columns or target_col not in train_df.columns:
        return {}
    rank_map = (
        train_df.groupby("Neighborhood")[target_col]
        .mean()
        .rank()
        .to_dict()
    )
    return rank_map
