# Leather Operations MLP Dashboard

This is a project I built to turn leather production data into something practical: a local dashboard + an MLP model for area forecasting.

The goal is simple:
- upload your Excel file
- see clean KPIs and trends
- train a model on your own data
- run predictions from the app

I originally built this around a retanning workflow, but it is now set up to work for any leather team with similar production columns.

## Quick Look

![Dashboard Preview](docs/dashboard-preview.svg)

## What It Does

- Local Excel upload directly in the app (`.xlsx`)
- Automatic English/Spanish column mapping (no manual column renaming needed in many cases)
- KPI dashboard: total area, total pieces, top families, top leather types, weekly trend
- MLP training from the currently loaded dataset
- MLP inference from family + leather type + piece count

## Exact Copy-Paste (Run Locally)

```bash
git clone https://github.com/js0596a/recurtido-mlp-dashboard.git
cd recurtido-mlp-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:8050`

If port `8050` is busy:

```bash
python -c "from app import app; app.run(host='127.0.0.1', port=8051, debug=True)"
```

## How To Use Your Own Data

1. Start the app.
2. Click **Upload Excel (.xlsx)**.
3. Pick your file.
4. Review filters/charts.
5. Click **Train model from current dataset**.
6. Run predictions in the MLP panel.

Everything runs locally on your machine.

## Reproducible CLI Workflow

Validate your file:

```bash
python mlp_recurtido.py validate --excel-path "/absolute/path/to/your_file.xlsx"
```

Train model artifacts:

```bash
python mlp_recurtido.py train --excel-path "/absolute/path/to/your_file.xlsx"
```

Single prediction from terminal:

```bash
python mlp_recurtido.py predict --familia "FAMILY_A" --tipo-cuero "LEATHER_A" --pzs 220
```

## Accepted Column Names

Canonical columns used internally:
- `FECHA`
- `TIPO DE CUERO`
- `FAMILIA`
- `PZS`
- `AREA TOTAL (ft2)`

The app/model also accept common English equivalents like:
- `DATE`
- `LEATHER TYPE`
- `FAMILY`
- `PIECES` / `QTY`
- `TOTAL AREA (ft2)`

See `DATA_SCHEMA.md` for details.

## Project Notes

- The repository name still includes `recurtido` for continuity, but UI/docs are now written in English-first terminology (`tanning` / `retanning`).
- Model/data artifacts are local and gitignored by default.
- No private company data is included in this repo.

## How The MLP Works With Your Company Data

This section explains exactly what happens after you upload your own Excel file and train the model.

1. You provide the training examples.
Each row in your dataset is treated as one historical production example, with:
- inputs: `TIPO DE CUERO`, `FAMILIA`, `PZS`
- target to learn: `AREA TOTAL (ft2)`

2. Column names are standardized automatically.
The app accepts Spanish and common English aliases (`PIEZAS`, `QTY`, `FAMILY`, `LEATHER TYPE`, etc.) and maps them to the canonical columns expected by the model.

3. Data is cleaned before training.
Rows are removed if required fields are missing, non-numeric, or invalid for learning (for example `PZS <= 0` or `AREA TOTAL (ft2) <= 0`).

4. Inputs are converted into machine-learning features.
`TIPO DE CUERO` and `FAMILIA` are one-hot encoded, and `PZS` is standardized with `StandardScaler`, so the neural network receives consistent numeric inputs.

5. The target area is also scaled.
`AREA TOTAL (ft2)` is standardized during training, then converted back to real units (`ft2`) after prediction.

6. The neural network (MLP) is trained.
Architecture:
- Dense(96, ReLU)
- Dropout(0.10)
- Dense(48, ReLU)
- Dense(16, ReLU)
- Dense(1) output for area regression

Training uses:
- train/test split
- early stopping (to avoid overfitting)
- learning-rate reduction on validation plateau

7. Model quality is measured and saved.
After training, the app reports `MAE`, `RMSE`, `MAPE`, and `R2`, plus row counts used for train/test.

8. Artifacts are stored locally for reuse.
The trained model, preprocessors, scaler, and metrics are saved under `artifacts/`, so predictions can be made later without retraining.

9. Prediction flow for new cases.
When a user enters `familia`, `tipo de cuero`, and `pzs`, the same preprocessing pipeline is applied, the model predicts area, and the result is converted back to `ft2`.

10. Yield is calculated from the prediction.
The dashboard computes:
`predicted_yield = predicted_area / pzs`

So the MLP is learning a production pattern from your own historical data and returning a practical forecast in the same business units your team uses.
