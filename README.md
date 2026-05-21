# Curtido / Recurtido MLP Dashboard

An employer-ready analytics app for **curtido/recurtido** production, combining:

- an interactive Dash dashboard for operational monitoring
- a neural network (MLP) for `AREA TOTAL (ft2)` forecasting

## What "Curtido" and "Recurtido" mean in this project

- **Curtido**: leather treatment process context
- **Recurtido**: retanning stage used for this model/dashboard scope

This project focuses on recurtido records and predicts expected total area from production attributes.

## Project goals

- monitor production behavior over time
- compare yield performance across families and leather types
- estimate `AREA TOTAL (ft2)` with an MLP model
- make the workflow reusable for anyone with similarly structured data

## How the MLP works (logic)

`mlp_recurtido.py` follows this pipeline:

1. Load Excel data and auto-detect the best worksheet (prefers `RECURTIDO` first).
2. Validate required columns:
   - `TIPO DE CUERO`
   - `FAMILIA`
   - `PZS`
   - `AREA TOTAL (ft2)`
3. Clean data:
   - normalize text fields (`strip`, `upper`, replace missing tokens)
   - coerce numeric fields and remove invalid rows (`PZS <= 0`, `AREA <= 0`)
4. Split into train/test sets (`80/20`).
5. Build preprocessing:
   - one-hot encoding for categorical features
   - standard scaling for numeric feature (`PZS`)
6. Scale target (`AREA TOTAL (ft2)`) with a separate scaler.
7. Train MLP regression network:
   - architecture: Dense(96) -> Dropout(0.10) -> Dense(48) -> Dense(16) -> Dense(1)
   - optimizer: Adam
   - loss: MSE
   - callbacks: EarlyStopping + ReduceLROnPlateau
8. Evaluate performance on held-out test data:
   - MAE
   - RMSE
   - MAPE
   - R2
9. Save artifacts (`artifacts/`) for later inference:
   - trained model (`.keras`)
   - preprocessor
   - target scaler
   - metrics dictionary

Inference (`predict_area_total`) applies the saved preprocessor and scaler to produce:

- predicted area (`ft2`)
- predicted yield (`ft2 / piece`)

## Tech stack

- Python
- Dash + Dash Mantine Components
- Plotly
- scikit-learn
- TensorFlow/Keras
- pandas / NumPy

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data requirements

Required columns:

- `FECHA`
- `TIPO DE CUERO`
- `FAMILIA`
- `PZS`
- `AREA TOTAL (ft2)`

The app/model tries `RECURTIDO` sheet first, then auto-detects a sheet with required columns.

See `/Users/jeslgdo/Documents/recurtido-mlp-dashboard/DATA_SCHEMA.md`.

## Bring your own data

Option 1. Place file at default path:

- `data/datosProd.xlsx`

Option 2. Set custom path and preferred sheet:

```bash
export RECURTIDO_EXCEL_PATH="/absolute/path/to/your_data.xlsx"
export RECURTIDO_SHEET_NAME="RECURTIDO"
```

## Validate data before training

```bash
python mlp_recurtido.py validate --excel-path "$RECURTIDO_EXCEL_PATH"
```

## Train the model

```bash
python mlp_recurtido.py train --excel-path "$RECURTIDO_EXCEL_PATH"
```

## Run dashboard

```bash
python app.py
```

Open: `http://127.0.0.1:8050`

If port 8050 is busy:

```bash
python -c "from app import app; app.run(host='127.0.0.1', port=8051, debug=True)"
```

## Quick CLI prediction

```bash
python mlp_recurtido.py predict --familia "XYZ" --tipo-cuero "ABC" --pzs 220
```

## Repository structure

```text
.
├── app.py
├── mlp_recurtido.py
├── DATA_SCHEMA.md
├── requirements.txt
├── assets/
│   └── styles.css
├── data/
│   └── .gitkeep
└── artifacts/
    └── .gitkeep
```

## Notes

- Raw datasets and artifacts are intentionally ignored by git to keep repo clean and reusable.
- Share code and workflow publicly; keep sensitive data local.
