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

When you upload your file and click train, I use your historical rows to learn this relationship:
- Inputs: `TIPO DE CUERO`, `FAMILIA`, `PZS`
- Target: `AREA TOTAL (ft2)`

Before training, I automatically map Spanish/English column names, clean invalid rows, one-hot encode categorical fields, and scale numeric values so the network sees stable inputs.

### Architecture (plain English)

1. `Dense(96, relu)`
First hidden layer. It starts learning broad interactions between leather type, family, and piece count.

2. `Dropout(0.10)`
During training, it randomly drops 10% of neurons each step, which helps reduce overfitting.

3. `Dense(48, relu)`
Second hidden layer. It refines the strongest patterns from layer 1.

4. `Dense(16, relu)`
Third hidden layer. It compresses the signal into a smaller, more focused representation.

5. `Dense(1)`
Output layer. It returns one number: predicted area (then converted back to real `ft2` units).

### Training behavior

- Optimizer: Adam (`learning_rate=0.001`)
- Loss: Mean Squared Error (`mse`)
- Validation split: 20% of training subset
- `EarlyStopping`: stops when validation loss no longer improves
- `ReduceLROnPlateau`: lowers learning rate when progress stalls

### Prediction output

For any new case (`familia`, `tipo de cuero`, `pzs`), the app applies the same preprocessing, predicts area in `ft2`, and computes:

`predicted_yield = predicted_area / pzs`

In short, the MLP is a compact feed-forward network that learns from your own production history and gives practical area/yield forecasts in business-friendly units.
