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

Short version:

1. You upload your Excel file.
2. I map your column names automatically (Spanish or English) to the required model fields.
3. I clean bad rows (missing values, non-numeric values, or rows where `PZS <= 0` or `AREA <= 0`).
4. I train the MLP to learn this relationship:
Inputs: `TIPO DE CUERO`, `FAMILIA`, `PZS`
Target: `AREA TOTAL (ft2)`
5. I evaluate model quality with `MAE`, `RMSE`, `MAPE`, and `R2`.
6. I save the model and preprocessors in `artifacts/` so you can reuse them.
7. For new inputs (`familia`, `tipo de cuero`, `pzs`), I predict area in `ft2` and compute yield:
`predicted_yield = predicted_area / pzs`

In simple terms: the model learns from your own historical production data and returns practical forecasts in the same business units your team uses.
