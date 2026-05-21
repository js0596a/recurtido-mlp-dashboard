# Recurtido MLP Dashboard

Production analytics + MLP forecasting app for leather recurtido operations.

## Privacy-first design

- Raw company data is **not** committed.
- Excel files under `data/` are gitignored.
- Trained artifacts under `artifacts/` are gitignored.
- This repo contains only code, docs, and reproducible workflow.

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

## Add your private data locally

Place your private Excel file locally (not tracked by git), for example:

- `data/datosProd.xlsx`

or set a custom path with env var:

```bash
export RECURTIDO_EXCEL_PATH="/absolute/private/path/datosProd.xlsx"
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
├── mlp_recurtido.py
├── requirements.txt
├── data/
│   └── .gitkeep
└── artifacts/
    └── .gitkeep
```
