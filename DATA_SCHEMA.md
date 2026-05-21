# Curtido / Recurtido Data Schema

The dashboard and model expect an Excel worksheet with these columns:

- `FECHA`
- `TIPO DE CUERO`
- `FAMILIA`
- `PZS`
- `AREA TOTAL (ft2)`

## Column expectations

- `FECHA`
  - date/datetime value
  - Excel serial dates are also accepted
- `TIPO DE CUERO`
  - categorical text
- `FAMILIA`
  - categorical text
- `PZS`
  - numeric, must be `> 0`
- `AREA TOTAL (ft2)`
  - numeric target, must be `> 0`

## Sheet selection logic

- Preferred sheet name is `RECURTIDO`.
- If that sheet is not valid, the code auto-detects another sheet containing all required columns.

## Typical usage

```bash
export RECURTIDO_EXCEL_PATH="/absolute/path/to/your_data.xlsx"
export RECURTIDO_SHEET_NAME="RECURTIDO"
python mlp_recurtido.py validate --excel-path "$RECURTIDO_EXCEL_PATH"
```
