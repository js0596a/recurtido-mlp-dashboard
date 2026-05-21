from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL_PATH = BASE_DIR / "data" / "datosProd.xlsx"
ARTIFACT_DIR = BASE_DIR / "artifacts"

FEATURES = ["TIPO DE CUERO", "FAMILIA", "PZS"]
TARGET = "AREA TOTAL (ft2)"
SHEET_NAME = "RETANNING"
LEGACY_SHEET_NAME = "RECURTIDO"
REQUIRED_COLUMNS = FEATURES + [TARGET]

COLUMN_ALIASES = {
    "TIPO DE CUERO": [
        "TIPO DE CUERO",
        "LEATHER TYPE",
        "TYPE OF LEATHER",
        "HIDE TYPE",
        "CUERO",
    ],
    "FAMILIA": [
        "FAMILIA",
        "FAMILY",
        "PRODUCT FAMILY",
        "PROCESS FAMILY",
        "CATEGORY",
    ],
    "PZS": [
        "PZS",
        "PCS",
        "PIECES",
        "PIEZAS",
        "UNITS",
        "QTY",
        "QUANTITY",
    ],
    "AREA TOTAL (ft2)": [
        "AREA TOTAL (ft2)",
        "TOTAL AREA (ft2)",
        "TOTAL AREA FT2",
        "TOTAL AREA",
        "AREA (ft2)",
        "AREA FT2",
        "SQFT",
        "SQUARE FEET",
    ],
}


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    preprocessor: Path
    y_scaler: Path
    metrics: Path


ARTIFACTS = ArtifactPaths(
    model=ARTIFACT_DIR / "retanning_mlp.keras",
    preprocessor=ARTIFACT_DIR / "retanning_preprocessor.joblib",
    y_scaler=ARTIFACT_DIR / "retanning_y_scaler.joblib",
    metrics=ARTIFACT_DIR / "retanning_metrics.joblib",
)

LEGACY_ARTIFACTS = ArtifactPaths(
    model=ARTIFACT_DIR / "recurtido_mlp.keras",
    preprocessor=ARTIFACT_DIR / "recurtido_preprocessor.joblib",
    y_scaler=ARTIFACT_DIR / "recurtido_y_scaler.joblib",
    metrics=ARTIFACT_DIR / "recurtido_metrics.joblib",
)


def _normalize_column_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_column_key(alias)] = canonical
        lookup[_normalize_column_key(canonical)] = canonical
    return lookup


ALIAS_LOOKUP = _alias_lookup()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map English/Spanish column variants to canonical model columns."""
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]

    rename_map: dict[str, str] = {}
    claimed: set[str] = set()

    for original_col in normalized.columns:
        canonical = ALIAS_LOOKUP.get(_normalize_column_key(original_col))
        if canonical and canonical not in claimed:
            rename_map[original_col] = canonical
            claimed.add(canonical)

    return normalized.rename(columns=rename_map)


def clean_retanning_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the expected fields and drop unusable rows."""
    standardized = standardize_column_names(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in standardized.columns]
    if missing:
        raise ValueError(
            "Missing required columns after alias mapping: "
            f"{missing}. Required canonical columns: {REQUIRED_COLUMNS}"
        )

    cleaned = standardized[REQUIRED_COLUMNS].copy()

    for cat_col in ["TIPO DE CUERO", "FAMILIA"]:
        cleaned[cat_col] = cleaned[cat_col].astype(str).str.strip().str.upper()
        cleaned[cat_col] = cleaned[cat_col].replace({"NAN": "NO DATA", "-": "NO DATA", "": "NO DATA"})

    cleaned["PZS"] = pd.to_numeric(cleaned["PZS"], errors="coerce")
    cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="coerce")

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned[(cleaned["PZS"] > 0) & (cleaned[TARGET] > 0)].copy()

    if cleaned.empty:
        raise ValueError("No valid rows remain after cleaning. Check numeric values and missing fields.")

    return cleaned


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_mlp(input_size: int, learning_rate: float = 0.001) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_size,)),
            layers.Dense(96, activation="relu"),
            layers.Dropout(0.10),
            layers.Dense(48, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), ["TIPO DE CUERO", "FAMILIA"]),
            ("numeric", StandardScaler(), ["PZS"]),
        ]
    )


def _ordered_sheet_candidates(sheet_names: list[str], preferred_sheet: str) -> list[str]:
    ordered = [preferred_sheet, SHEET_NAME, LEGACY_SHEET_NAME]
    ordered.extend(sheet_names)

    seen: set[str] = set()
    candidates: list[str] = []
    for name in ordered:
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    return candidates


def load_training_sheet(excel_path: Path, preferred_sheet: str = SHEET_NAME) -> tuple[pd.DataFrame, str]:
    """
    Load preferred sheet when available; otherwise auto-detect the first
    worksheet containing required model columns (with English/Spanish aliases).
    """
    try:
        excel_book = pd.ExcelFile(excel_path)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    sheet_names = excel_book.sheet_names
    candidates = _ordered_sheet_candidates(sheet_names, preferred_sheet)

    for sheet in candidates:
        if sheet not in sheet_names:
            continue
        raw = pd.read_excel(excel_path, sheet_name=sheet)
        standardized = standardize_column_names(raw)
        if all(col in standardized.columns for col in REQUIRED_COLUMNS):
            return standardized, sheet

    raise ValueError(
        "No worksheet contains required model columns "
        f"{REQUIRED_COLUMNS}. Available sheets: {sheet_names}"
    )


def _save_artifacts(
    model: keras.Model,
    preprocessor: ColumnTransformer,
    y_scaler: StandardScaler,
    metrics: dict,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(ARTIFACTS.model))
    joblib.dump(preprocessor, ARTIFACTS.preprocessor)
    joblib.dump(y_scaler, ARTIFACTS.y_scaler)
    joblib.dump(metrics, ARTIFACTS.metrics)
    load_artifacts.cache_clear()


def _train_with_clean_df(
    cleaned_df: pd.DataFrame,
    sheet_label: str,
    test_size: float,
    random_state: int,
) -> dict:
    if len(cleaned_df) < 30:
        raise ValueError(
            "At least 30 valid rows are recommended for training. "
            f"Found {len(cleaned_df)} rows after cleaning."
        )

    tf.keras.utils.set_random_seed(random_state)

    X = cleaned_df[FEATURES]
    y = cleaned_df[[TARGET]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = _build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    model = build_mlp(X_train_processed.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        X_train_processed,
        y_train_scaled,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    predictions_scaled = model.predict(X_test_processed, verbose=0)
    predictions = y_scaler.inverse_transform(predictions_scaled).ravel()
    predictions = np.maximum(predictions, 0)

    y_test_values = y_test[TARGET].values

    mae = mean_absolute_error(y_test_values, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_values, predictions))
    r2 = r2_score(y_test_values, predictions)
    mape = float(np.mean(np.abs((y_test_values - predictions) / np.maximum(y_test_values, 1e-8))) * 100)

    metrics = {
        "rows_used": int(len(cleaned_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_pct": float(mape),
        "r2": float(r2),
        "epochs_trained": int(len(history.history["loss"])),
        "sheet_name": sheet_label,
        "features": FEATURES,
        "target": TARGET,
    }

    _save_artifacts(model=model, preprocessor=preprocessor, y_scaler=y_scaler, metrics=metrics)
    return metrics


def train_and_save_model(
    excel_path: Path | str = DEFAULT_EXCEL_PATH,
    sheet_name: str = SHEET_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found at {excel_path}. Set --excel-path to your local file."
        )

    raw_df, selected_sheet = load_training_sheet(excel_path=excel_path, preferred_sheet=sheet_name)
    cleaned_df = clean_retanning_data(raw_df)
    return _train_with_clean_df(
        cleaned_df=cleaned_df,
        sheet_label=selected_sheet,
        test_size=test_size,
        random_state=random_state,
    )


def train_and_save_from_dataframe(
    dataframe: pd.DataFrame,
    source_label: str = "uploaded-dataset",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    cleaned_df = clean_retanning_data(dataframe)
    return _train_with_clean_df(
        cleaned_df=cleaned_df,
        sheet_label=source_label,
        test_size=test_size,
        random_state=random_state,
    )


def _choose_artifact_set() -> ArtifactPaths | None:
    current_files = [ARTIFACTS.model, ARTIFACTS.preprocessor, ARTIFACTS.y_scaler, ARTIFACTS.metrics]
    legacy_files = [LEGACY_ARTIFACTS.model, LEGACY_ARTIFACTS.preprocessor, LEGACY_ARTIFACTS.y_scaler, LEGACY_ARTIFACTS.metrics]

    if all(path.exists() for path in current_files):
        return ARTIFACTS
    if all(path.exists() for path in legacy_files):
        return LEGACY_ARTIFACTS
    return None


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[keras.Model, ColumnTransformer, StandardScaler, dict]:
    artifact_set = _choose_artifact_set()

    if artifact_set is None:
        missing_str = ", ".join(str(p) for p in [ARTIFACTS.model, ARTIFACTS.preprocessor, ARTIFACTS.y_scaler, ARTIFACTS.metrics])
        raise FileNotFoundError(
            "Model artifacts not found. Run training first: "
            "python mlp_recurtido.py train --excel-path /path/to/your_data.xlsx. "
            f"Expected files: {missing_str}"
        )

    model = keras.models.load_model(str(artifact_set.model))
    preprocessor = joblib.load(artifact_set.preprocessor)
    y_scaler = joblib.load(artifact_set.y_scaler)
    metrics = joblib.load(artifact_set.metrics)

    return model, preprocessor, y_scaler, metrics


def predict_area_total(familia: str, tipo_cuero: str, pzs: float | int) -> dict:
    if pzs is None or float(pzs) <= 0:
        raise ValueError("pzs must be > 0")

    model, preprocessor, y_scaler, metrics = load_artifacts()

    input_df = pd.DataFrame(
        [
            {
                "TIPO DE CUERO": str(tipo_cuero).strip().upper(),
                "FAMILIA": str(familia).strip().upper(),
                "PZS": float(pzs),
            }
        ]
    )

    input_processed = preprocessor.transform(input_df)
    prediction_scaled = model.predict(input_processed, verbose=0)
    prediction = y_scaler.inverse_transform(prediction_scaled)[0][0]

    predicted_area = max(float(prediction), 0.0)
    predicted_yield = predicted_area / float(pzs)

    return {
        "predicted_area": predicted_area,
        "predicted_rendimiento": predicted_yield,
        "metrics": metrics,
    }


def _print_metrics(metrics: dict) -> None:
    print("Training complete")
    print(json.dumps(metrics, indent=2))


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train and use an MLP model for leather retanning area forecasting."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train model and save artifacts")
    train_parser.add_argument("--excel-path", default=str(DEFAULT_EXCEL_PATH), help="Path to local Excel file")
    train_parser.add_argument(
        "--sheet",
        default=SHEET_NAME,
        help="Preferred sheet name (auto-detect fallback checks RETANNING/RECURTIDO and all sheets)",
    )

    validate_parser = sub.add_parser("validate", help="Validate that dataset has required columns")
    validate_parser.add_argument("--excel-path", default=str(DEFAULT_EXCEL_PATH), help="Path to local Excel file")
    validate_parser.add_argument("--sheet", default=SHEET_NAME, help="Preferred sheet name")

    pred_parser = sub.add_parser("predict", help="Run one prediction with saved artifacts")
    pred_parser.add_argument("--familia", required=True)
    pred_parser.add_argument("--tipo-cuero", required=True)
    pred_parser.add_argument("--pzs", required=True, type=float)

    args = parser.parse_args()

    if args.command == "train":
        metrics = train_and_save_model(excel_path=args.excel_path, sheet_name=args.sheet)
        _print_metrics(metrics)
    elif args.command == "validate":
        dataset, selected_sheet = load_training_sheet(excel_path=Path(args.excel_path), preferred_sheet=args.sheet)
        cleaned = clean_retanning_data(dataset)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "selected_sheet": selected_sheet,
                    "rows_raw": int(len(dataset)),
                    "rows_after_cleaning": int(len(cleaned)),
                    "columns_detected": [str(col).strip() for col in dataset.columns],
                    "required_columns": REQUIRED_COLUMNS,
                },
                indent=2,
            )
        )
    elif args.command == "predict":
        result = predict_area_total(familia=args.familia, tipo_cuero=args.tipo_cuero, pzs=args.pzs)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
