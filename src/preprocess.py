from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_COLUMNS: List[str] = [
    "sleep_hours",
    "stress_level",
    "work_hours",
    "social_activity",
    "physical_activity",
    "screen_time",
]
TARGET_COLUMN = "risk_level"
VALID_RISK_LEVELS = ["Low", "Medium", "High"]

COLUMN_ALIASES = {
    "sleep_hours": ["sleep_hours", "Sleep Hours"],
    "stress_level": ["stress_level", "Stress Level"],
    "work_hours": ["work_hours", "Work Hours per Week"],
    "social_activity": ["social_activity", "Social Interaction Score"],
    "physical_activity": ["physical_activity", "Exercise Level"],
    "screen_time": ["screen_time", "Screen Time per Day (Hours)"],
}


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load CSV data from disk."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def map_and_standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map known alternate headers to canonical feature names."""
    standardized = df.copy()
    rename_map = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        existing = first_existing_column(standardized, aliases)
        if existing is None:
            raise ValueError(
                f"Missing required feature column for '{canonical_name}'. "
                f"Accepted names: {aliases}"
            )
        rename_map[existing] = canonical_name

    standardized = standardized.rename(columns=rename_map)

    # Convert exercise level categories into numeric activity scale if needed.
    if standardized["physical_activity"].dtype == object:
        exercise_map = {
            "low": 2.0,
            "moderate": 5.0,
            "medium": 5.0,
            "high": 8.0,
            "very high": 10.0,
        }
        standardized["physical_activity"] = (
            standardized["physical_activity"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(exercise_map)
        )

    if standardized["stress_level"].dtype == object:
        stress_map = {
            "low": 3.0,
            "moderate": 6.0,
            "medium": 6.0,
            "high": 9.0,
        }
        standardized["stress_level"] = (
            standardized["stress_level"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(stress_map)
        )

    for feature in FEATURE_COLUMNS:
        standardized[feature] = pd.to_numeric(standardized[feature], errors="coerce")

    return standardized


def normalize_risk_labels(series: pd.Series) -> pd.Series:
    """Normalize target labels so values are consistently Low/Medium/High."""
    normalized = series.astype(str).str.strip().str.capitalize()
    return normalized


def derive_risk_level(df: pd.DataFrame) -> pd.Series:
    """Create Low/Medium/High risk labels using explicit lifestyle rules."""
    features = df[FEATURE_COLUMNS].copy()
    for feature in FEATURE_COLUMNS:
        features[feature] = pd.to_numeric(features[feature], errors="coerce")
        features[feature] = features[feature].fillna(features[feature].median())

    sleep = features["sleep_hours"]
    stress = features["stress_level"]
    work = features["work_hours"]
    social = features["social_activity"]
    physical = features["physical_activity"]
    screen = features["screen_time"]

    labels = pd.Series("Medium", index=df.index, dtype="object")

    risk_points = (
        (stress >= 8).astype(int) * 2
        + ((stress >= 6) & (stress < 8)).astype(int)
        + (sleep < 5).astype(int) * 2
        + ((sleep >= 5) & (sleep < 6)).astype(int)
        + (work > 50).astype(int)
        + (screen > 7).astype(int)
        + (social < 4).astype(int)
        + (physical < 4).astype(int)
    )

    protective_points = (
        ((sleep >= 7) & (sleep <= 9)).astype(int)
        + (stress <= 4).astype(int)
        + (work <= 45).astype(int)
        + (screen <= 5).astype(int)
        + (social >= 6).astype(int)
        + (physical >= 6).astype(int)
    )

    # High risk: strong stress + poor recovery profile.
    high_risk_mask = (
        ((stress > 7) & (sleep < 5.5))
        | ((stress > 8) & ((screen > 8) | (work > 55)))
        | ((sleep < 4.5) & ((social < 3.5) | (physical < 3.5)))
        | (risk_points >= 5)
    )

    # Low risk: healthy and balanced lifestyle pattern.
    low_risk_mask = (
        (protective_points >= 5)
        & (risk_points <= 1)
    )

    labels.loc[high_risk_mask] = "High"
    labels.loc[~high_risk_mask & low_risk_mask] = "Low"
    return labels


def resolve_target(df: pd.DataFrame) -> pd.Series:
    """Use risk_level if available, otherwise derive a risk label."""
    if TARGET_COLUMN in df.columns:
        candidate = normalize_risk_labels(df[TARGET_COLUMN])
        if set(candidate.dropna().unique()).issubset(set(VALID_RISK_LEVELS)):
            return candidate

    # If the dataset has no risk_level column, generate one for supervised learning.
    return derive_risk_level(df)


def build_preprocessor() -> ColumnTransformer:
    """Build numeric preprocessing: median imputation + standard scaling."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, FEATURE_COLUMNS)],
        remainder="drop",
    )


def prepare_data(
    data_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, LabelEncoder]:
    """Load, clean, encode target, and split the dataset."""
    raw_df = load_dataset(data_path)
    df = map_and_standardize_features(raw_df)
    df[TARGET_COLUMN] = resolve_target(df)

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    invalid_labels = sorted(set(df[TARGET_COLUMN].unique()) - set(VALID_RISK_LEVELS))
    if invalid_labels:
        raise ValueError(f"Target labels must be one of {VALID_RISK_LEVELS}. Got: {invalid_labels}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    preprocessor = build_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor, label_encoder
