# Recurtido MLP Dashboard

Production analytics + MLP forecasting app for leather recurtido operations.

## Bring your own data workflow

Any user can run this project with their own Excel dataset.

- Put your `.xlsx` file locally in `data/` or anywhere on your machine.
- Train the MLP with the CLI.
- Launch the dashboard and run predictions.

This repo intentionally tracks code only. Dataset files and trained artifacts are excluded via `.gitignore`.

## What this project includes

- `app.py`: interactive Dash dashboard (KPIs, filters, charts, model inference panel)
- `mlp_recurtido.py`: train/predict pipeline for `AREA TOTAL (ft2)` using `TIPO DE CUERO`, `FAMILIA`, and `PZS`
- CLI for training and single prediction checks

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset format

Required columns:

- `FECHA`
- `TIPO DE CUERO`
- `FAMILIA`
- `PZS`
- `AREA TOTAL (ft2)`

The code will use preferred sheet `RECURTIDO` first, then auto-detect another sheet containing required columns.
See `/DATA_SCHEMA.md` for quick reference.

## Point to your file

Option 1: Put file at default path:

- `data/datosProd.xlsx`

Option 2: Use environment variables:

```bash
export RECURTIDO_EXCEL_PATH="/absolute/private/path/datosProd.xlsx"
export RECURTIDO_SHEET_NAME="RECURTIDO"
```

## Validate data before training

```bash
python mlp_recurtido.py validate --excel-path "$RECURTIDO_EXCEL_PATH"
```

## Train model

```bash
python mlp_recurtido.py train --excel-path "$RECURTIDO_EXCEL_PATH"
```

If your file is already in `data/datosProd.xlsx`, you can run:

```bash
python mlp_recurtido.py train
```

## Run dashboard

```bash
python app.py
```

Open: `http://127.0.0.1:8050`

## Single prediction from terminal

```bash
python mlp_recurtido.py predict --familia "XYZ" --tipo-cuero "ABC" --pzs 220
```

## Repo structure

```text
.
├── app.py
├── DATA_SCHEMA.md
├── mlp_recurtido.py
├── requirements.txt
├── data/
│   └── .gitkeep
└── artifacts/
    └── .gitkeep
```
