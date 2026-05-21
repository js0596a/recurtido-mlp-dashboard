# Data Schema

Use an Excel file that includes these columns:

- `FECHA`
- `TIPO DE CUERO`
- `FAMILIA`
- `PZS`
- `AREA TOTAL (ft2)`

## Notes

- `FECHA` can be Excel serial date or normal datetime string.
- `PZS` and `AREA TOTAL (ft2)` must be numeric and greater than 0.
- The app/model will prefer sheet `RECURTIDO`, but auto-detect another sheet if it contains required columns.
