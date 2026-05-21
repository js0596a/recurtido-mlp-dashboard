from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import json

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
SHEET_NAME = "RECURTIDO"
REQUIRED_COLUMNS = FEATURES + [TARGET]


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path = ARTIFACT_DIR / "recurtido_mlp.keras"
    preprocessor: Path = ARTIFACT_DIR / "recurtido_preprocessor.joblib"
    y_scaler: Path = ARTIFACT_DIR / "recurtido_y_scaler.joblib"
    metrics: Path = ARTIFACT_DIR / "recurtido_metrics.joblib"


ARTIFACTS = ArtifactPaths()


def clean_recurtido_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the expected fields and drop unusable rows."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df[REQUIRED_COLUMNS].copy()

    for cat_col in ["TIPO DE CUERO", "FAMILIA"]:
        cleaned[cat_col] = cleaned[cat_col].astype(str).str.strip().str.upper()
        cleaned[cat_col] = cleaned[cat_col].replace({"NAN": "SIN DATO", "-": "SIN DATO", "": "SIN DATO"})

    cleaned["PZS"] = pd.to_numeric(cleaned["PZS"], errors="coerce")
    cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="coerce")

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned[(cleaned["PZS"] > 0) & (cleaned[TARGET] > 0)].copy()

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


def load_training_sheet(excel_path: Path, preferred_sheet: str = SHEET_NAME) -> tuple[pd.DataFrame, str]:
    """
    Load the preferred sheet if available, otherwise auto-detect the first sheet
    containing all required training columns.
    """
    try:
        excel_book = pd.ExcelFile(excel_path)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    sheet_names = excel_book.sheet_names

    candidate_sheets = [preferred_sheet] + [name for name in sheet_names if name != preferred_sheet]

    for sheet in candidate_sheets:
        if sheet not in sheet_names:
            continue
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df.columns = [str(col).strip() for col in df.columns]
        if all(col in df.columns for col in REQUIRED_COLUMNS):
            return df, sheet

    raise ValueError(
        "No worksheet contains the required columns "
        f"{REQUIRED_COLUMNS}. Available sheets: {sheet_names}"
    )


def train_and_save_model(
    excel_path: Path | str = DEFAULT_EXCEL_PATH,
    sheet_name: str = SHEET_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found at {excel_path}. Set --excel-path to your private local file."
        )

    tf.keras.utils.set_random_seed(random_state)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, selected_sheet = load_training_sheet(excel_path=excel_path, preferred_sheet=sheet_name)
    df = clean_recurtido_data(raw_df)

    X = df[FEATURES]
    y = df[[TARGET]]

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
    y_test_scaled = y_scaler.transform(y_test)

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
        verbose=1,
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
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_pct": float(mape),
        "r2": float(r2),
        "epochs_trained": int(len(history.history["loss"])),
        "sheet_name": selected_sheet,
        "features": FEATURES,
        "target": TARGET,
    }

    model.save(str(ARTIFACTS.model))
    joblib.dump(preprocessor, ARTIFACTS.preprocessor)
    joblib.dump(y_scaler, ARTIFACTS.y_scaler)
    joblib.dump(metrics, ARTIFACTS.metrics)

    return metrics


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[keras.Model, ColumnTransformer, StandardScaler, dict]:
    missing = [
        path
        for path in [ARTIFACTS.model, ARTIFACTS.preprocessor, ARTIFACTS.y_scaler, ARTIFACTS.metrics]
        if not path.exists()
    ]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Model artifacts not found. Run training first: "
            "python mlp_recurtido.py train --excel-path /private/path/datosProd.xlsx. "
            f"Missing: {missing_str}"
        )

    model = keras.models.load_model(str(ARTIFACTS.model))
    preprocessor = joblib.load(ARTIFACTS.preprocessor)
    y_scaler = joblib.load(ARTIFACTS.y_scaler)
    metrics = joblib.load(ARTIFACTS.metrics)

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
    predicted_rendimiento = predicted_area / float(pzs)

    return {
        "predicted_area": predicted_area,
        "predicted_rendimiento": predicted_rendimiento,
        "metrics": metrics,
    }


def _print_metrics(metrics: dict) -> None:
    print("Training complete")
    print(json.dumps(metrics, indent=2))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train and use MLP model for recurtido area forecasting.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train model and save artifacts")
    train_parser.add_argument("--excel-path", default=str(DEFAULT_EXCEL_PATH), help="Path to local Excel file")
    train_parser.add_argument("--sheet", default=SHEET_NAME, help="Preferred sheet name (auto-detect fallback enabled)")

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
        print(
            json.dumps(
                {
                    "status": "ok",
                    "selected_sheet": selected_sheet,
                    "rows": int(len(dataset)),
                    "columns": [str(col).strip() for col in dataset.columns],
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
