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

## How the MLP works with your data

When you upload your own Excel file and click train, this is what happens under the hood.

1. I treat each row as one historical production example.
The model learns this relationship:
- Inputs: `TIPO DE CUERO`, `FAMILIA`, `PZS`
- Target: `AREA TOTAL (ft2)`

2. I normalize column names automatically.
You can use Spanish or English headers (`PIEZAS`, `QTY`, `FAMILY`, `LEATHER TYPE`, etc.) and the app maps them to the model's canonical columns.

3. I clean the data before training.
Rows are removed if required values are missing, non-numeric, or not usable for learning (for example `PZS <= 0` or `AREA TOTAL (ft2) <= 0`).

4. I convert inputs into model-ready features.
`TIPO DE CUERO` and `FAMILIA` are one-hot encoded, and `PZS` is scaled with `StandardScaler`.

5. I scale the target area for training stability.
`AREA TOTAL (ft2)` is scaled during training and converted back to real `ft2` units after prediction.

6. I train a feed-forward MLP for regression.
Network architecture:
- Dense(96, ReLU)
- Dropout(0.10)
- Dense(48, ReLU)
- Dense(16, ReLU)
- Dense(1) output

Training setup:
- train/test split
- early stopping
- learning-rate reduction on validation plateau

7. I evaluate and save quality metrics.
After training, the app stores `MAE`, `RMSE`, `MAPE`, and `R2`, along with train/test row counts.

8. I save all artifacts locally.
Model, preprocessors, scaler, and metrics are written to `artifacts/` so you can predict later without retraining.

9. For a new case, I run the same preprocessing pipeline.
Given `familia`, `tipo de cuero`, and `pzs`, the model predicts area and returns it in `ft2`.

10. Yield is calculated from the predicted area.
`predicted_yield = predicted_area / pzs`

In short, the model learns from your own production history and gives you forecasts in the same units your team already uses.
