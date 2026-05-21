# Leather Data Schema

The dashboard and model expect these canonical fields:

- `FECHA`
- `TIPO DE CUERO`
- `FAMILIA`
- `PZS`
- `AREA TOTAL (ft2)`

## English/Spanish Alias Support

You can also use common English names. The pipeline maps aliases automatically.

Examples:

- `FECHA`: `DATE`, `PRODUCTION DATE`, `DAY`
- `TIPO DE CUERO`: `LEATHER TYPE`, `TYPE OF LEATHER`, `HIDE TYPE`
- `FAMILIA`: `FAMILY`, `PRODUCT FAMILY`, `CATEGORY`
- `PZS`: `PIECES`, `PCS`, `QTY`, `QUANTITY`
- `AREA TOTAL (ft2)`: `TOTAL AREA (ft2)`, `TOTAL AREA`, `AREA FT2`, `SQFT`

## Column Expectations

- `FECHA`
  - date/datetime value
  - Excel serial dates are accepted
- `TIPO DE CUERO`
  - categorical text
- `FAMILIA`
  - categorical text
- `PZS`
  - numeric, must be `> 0`
- `AREA TOTAL (ft2)`
  - numeric target, must be `> 0`

## Sheet Selection Logic

Preferred sheet name is `RETANNING`.

Fallback behavior:
- tries `RECURTIDO` next
- then checks every sheet and picks the first one containing all required columns

## Typical Usage

```bash
python mlp_recurtido.py validate --excel-path "/absolute/path/to/your_file.xlsx"
python mlp_recurtido.py train --excel-path "/absolute/path/to/your_file.xlsx"
```
