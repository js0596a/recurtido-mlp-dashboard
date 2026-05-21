from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path

import dash
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from mlp_recurtido import predict_area_total, train_and_save_from_dataframe
    MLP_IMPORT_ERROR = None
except Exception as exc:
    predict_area_total = None
    train_and_save_from_dataframe = None
    MLP_IMPORT_ERROR = str(exc)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path(
    os.getenv("LEATHER_EXCEL_PATH", os.getenv("RECURTIDO_EXCEL_PATH", BASE_DIR / "data" / "datosProd.xlsx"))
)
PREFERRED_SHEET = os.getenv("LEATHER_SHEET_NAME", os.getenv("RECURTIDO_SHEET_NAME", "RETANNING"))
REPO_URL = "https://github.com/js0596a/recurtido-mlp-dashboard"
REQUIRED_COLUMNS = ["FECHA", "TIPO DE CUERO", "FAMILIA", "PZS", "AREA TOTAL (ft2)"]
METRICS_PATHS = [
    BASE_DIR / "artifacts" / "retanning_metrics.joblib",
    BASE_DIR / "artifacts" / "recurtido_metrics.joblib",
]

COLUMN_ALIASES = {
    "FECHA": [
        "FECHA",
        "DATE",
        "PRODUCTION DATE",
        "DAY",
        "TIMESTAMP",
    ],
    "TIPO DE CUERO": [
        "TIPO DE CUERO",
        "LEATHER TYPE",
        "TYPE OF LEATHER",
        "HIDE TYPE",
        "CUERO",
    ],
    "FAMILIA": [
        "FAMILIA",
        "FAMILY",
        "PRODUCT FAMILY",
        "PROCESS FAMILY",
        "CATEGORY",
    ],
    "PZS": [
        "PZS",
        "PCS",
        "PIECES",
        "PIEZAS",
        "UNITS",
        "QTY",
        "QUANTITY",
    ],
    "AREA TOTAL (ft2)": [
        "AREA TOTAL (ft2)",
        "TOTAL AREA (ft2)",
        "TOTAL AREA FT2",
        "TOTAL AREA",
        "AREA (ft2)",
        "AREA FT2",
        "SQFT",
        "SQUARE FEET",
    ],
}


def _normalize_column_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_column_key(alias)] = canonical
        lookup[_normalize_column_key(canonical)] = canonical
    return lookup


ALIAS_LOOKUP = _alias_lookup()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]

    rename_map: dict[str, str] = {}
    claimed: set[str] = set()

    for col in normalized.columns:
        canonical = ALIAS_LOOKUP.get(_normalize_column_key(col))
        if canonical and canonical not in claimed:
            rename_map[col] = canonical
            claimed.add(canonical)

    return normalized.rename(columns=rename_map)


def fix_excel_date(column: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(column):
        return column

    numeric_dates = pd.to_numeric(column, errors="coerce")
    if numeric_dates.notna().mean() > 0.8:
        return pd.to_datetime(numeric_dates, unit="D", origin="1899-12-30", errors="coerce")

    return pd.to_datetime(column, errors="coerce")


def _ordered_sheet_candidates(sheet_names: list[str], preferred_sheet: str) -> list[str]:
    ordered = [preferred_sheet, "RETANNING", "RECURTIDO"]
    ordered.extend(sheet_names)

    seen: set[str] = set()
    result: list[str] = []
    for name in ordered:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def select_sheet_with_columns(excel_book: pd.ExcelFile, preferred_sheet: str) -> tuple[pd.DataFrame, str]:
    sheet_names = excel_book.sheet_names
    candidates = _ordered_sheet_candidates(sheet_names, preferred_sheet)

    for sheet in candidates:
        if sheet not in sheet_names:
            continue
        candidate_df = pd.read_excel(excel_book, sheet_name=sheet)
        candidate_df = standardize_column_names(candidate_df)
        if all(col in candidate_df.columns for col in REQUIRED_COLUMNS):
            return candidate_df, sheet

    raise ValueError(
        "No worksheet contains required columns "
        f"{REQUIRED_COLUMNS}. Available sheets: {sheet_names}"
    )


def clean_dashboard_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    standardized = standardize_column_names(raw_df)
    missing = [col for col in REQUIRED_COLUMNS if col not in standardized.columns]
    if missing:
        raise ValueError(
            "Missing required columns after alias mapping: "
            f"{missing}. Required canonical columns: {REQUIRED_COLUMNS}"
        )

    df = standardized[REQUIRED_COLUMNS].copy()
    df["FECHA"] = fix_excel_date(df["FECHA"])

    df["TIPO DE CUERO"] = df["TIPO DE CUERO"].astype(str).str.strip().str.upper()
    df["FAMILIA"] = df["FAMILIA"].astype(str).str.strip().str.upper()

    df["PZS"] = pd.to_numeric(df["PZS"], errors="coerce")
    df["AREA TOTAL (ft2)"] = pd.to_numeric(df["AREA TOTAL (ft2)"], errors="coerce")

    df = df.dropna(subset=["FECHA", "PZS", "AREA TOTAL (ft2)"])

    df["TIPO DE CUERO"] = df["TIPO DE CUERO"].replace({"NAN": "NO DATA", "-": "NO DATA", "": "NO DATA"})
    df["FAMILIA"] = df["FAMILIA"].replace({"NAN": "NO DATA", "-": "NO DATA", "": "NO DATA"})

    df = df[(df["PZS"] > 0) & (df["AREA TOTAL (ft2)"] > 0)].copy()
    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Check date, pieces, and total area values.")

    df["SEMANA"] = df["FECHA"].dt.to_period("W").dt.start_time
    return df


def load_dashboard_data(excel_path: Path) -> tuple[pd.DataFrame, str]:
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found at {excel_path}. Upload a file in the app or set LEATHER_EXCEL_PATH."
        )

    try:
        excel_book = pd.ExcelFile(excel_path)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    raw_df, selected_sheet = select_sheet_with_columns(excel_book, preferred_sheet=PREFERRED_SHEET)
    return clean_dashboard_data(raw_df), selected_sheet


def load_dashboard_data_from_bytes(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    try:
        excel_book = pd.ExcelFile(io.BytesIO(file_bytes))
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    raw_df, selected_sheet = select_sheet_with_columns(excel_book, preferred_sheet=PREFERRED_SHEET)
    return clean_dashboard_data(raw_df), selected_sheet


def serialize_df(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso")


def deserialize_df(data_json: str | None) -> pd.DataFrame:
    if not data_json:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["SEMANA"])

    df = pd.read_json(io.StringIO(data_json), orient="split")
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    if "SEMANA" in df.columns:
        df["SEMANA"] = pd.to_datetime(df["SEMANA"], errors="coerce")
    return df


def make_dataset_meta(
    status: str,
    source: str,
    message: str,
    sheet: str | None = None,
    rows: int = 0,
    filename: str | None = None,
) -> dict:
    return {
        "status": status,
        "source": source,
        "message": message,
        "sheet": sheet,
        "rows": int(rows),
        "filename": filename,
    }


def load_saved_metrics() -> dict | None:
    if joblib is None:
        return None

    for path in METRICS_PATHS:
        if path.exists():
            try:
                metrics = joblib.load(path)
                if isinstance(metrics, dict):
                    return metrics
            except Exception:
                continue
    return None


def empty_chart(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=18, r=18, t=48, b=18),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def style_chart(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        margin=dict(l=18, r=18, t=48, b=18),
        hovermode="x unified",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="'Space Grotesk', Inter, sans-serif", color="#0f172a"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def make_mlp_model_badge(metrics: dict | None) -> dmc.Badge:
    if metrics:
        return dmc.Badge(
            f"MLP trained (R2 {metrics.get('r2', 0):.3f})",
            color="teal",
            variant="light",
            radius="sm",
        )
    return dmc.Badge("MLP not trained yet", color="orange", variant="light", radius="sm")


def compute_filter_options(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, list[str], list[str], list[str]]:
    if df.empty:
        today = pd.Timestamp.today().normalize()
        return today, today, ["ALL"], ["NO DATA"], ["NO DATA"]

    min_date = df["FECHA"].min()
    max_date = df["FECHA"].max()
    leather_options = ["ALL"] + sorted(df["TIPO DE CUERO"].dropna().unique())
    family_options = sorted(df["FAMILIA"].dropna().unique())
    mlp_leather_options = sorted(df["TIPO DE CUERO"].dropna().unique())
    return min_date, max_date, leather_options, family_options, mlp_leather_options


def compute_week_options(df: pd.DataFrame) -> tuple[list[dict], str | None]:
    if df.empty or "SEMANA" not in df.columns:
        return [], None

    weeks = sorted(pd.to_datetime(df["SEMANA"].dropna().unique()))
    options = [{"label": pd.Timestamp(week).strftime("%Y-%m-%d"), "value": pd.Timestamp(week).strftime("%Y-%m-%d")} for week in weeks]
    value = options[-1]["value"] if options else None
    return options, value


def make_mix_waterfall_figure(
    previous_yield: float,
    mix_effect: float,
    execution_effect: float,
    interaction_effect: float,
    current_yield: float,
) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Prev Week", "Mix Effect", "Execution Effect", "Interaction", "Current Week"],
            y=[previous_yield, mix_effect, execution_effect, interaction_effect, current_yield],
            connector={"line": {"color": "rgba(100,116,139,0.5)"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#0f172a"}},
        )
    )
    fig.update_yaxes(title="Yield (ft2 per piece)")
    return style_chart(fig, "Why Yield Changed (Week vs Previous Week)")


def make_mix_share_figure(merged: pd.DataFrame, prev_label: str, curr_label: str) -> go.Figure:
    ranked = merged.assign(total_pzs=merged["pzs_prev"] + merged["pzs_curr"]).sort_values("total_pzs", ascending=False).head(10)
    if ranked.empty:
        return empty_chart("Family Mix Shift", "Not enough family data to compare weeks.")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked["FAMILIA"],
            y=ranked["share_prev"] * 100,
            name=f"{prev_label} share",
            marker=dict(color="#94a3b8"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=ranked["FAMILIA"],
            y=ranked["share_curr"] * 100,
            name=f"{curr_label} share",
            marker=dict(color="#0ea5a4"),
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Family")
    fig.update_yaxes(title="Share of pieces (%)")
    return style_chart(fig, "Family Mix Shift (Pieces Share)")


def make_mix_driver_scatter(merged: pd.DataFrame, previous_yield: float) -> go.Figure:
    if merged.empty:
        return empty_chart("Family Impact Map", "No family contribution rows available.")

    point_size = (merged["pzs_curr"].fillna(0) / max(merged["pzs_curr"].max(), 1)) * 34 + 10
    reference_yield = merged["yield_prev"].where(merged["yield_prev"] > 0, merged["yield_curr"])

    fig = go.Figure(
        data=[
            go.Scatter(
                x=reference_yield,
                y=merged["share_delta_pct"],
                mode="markers+text",
                text=merged["FAMILIA"],
                textposition="top center",
                marker=dict(
                    size=point_size,
                    color=merged["mix_contrib"],
                    colorscale=[
                        [0.0, "#ef4444"],
                        [0.5, "#f59e0b"],
                        [1.0, "#22c55e"],
                    ],
                    line=dict(width=1, color="rgba(15,23,42,0.22)"),
                    showscale=True,
                    colorbar=dict(title="Mix contribution"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Reference yield: %{x:.2f} ft2/piece<br>"
                    "Share delta: %{y:.2f} pp<extra></extra>"
                ),
            )
        ]
    )
    fig.add_vline(x=previous_yield, line_dash="dash", line_color="#475569")
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b")
    fig.update_xaxes(title="Family reference yield (ft2/piece)")
    fig.update_yaxes(title="Share change (percentage points)")
    return style_chart(fig, "Family Impact Map (Low-Yield Mix Increase)")


try:
    DEFAULT_DF, DEFAULT_SHEET = load_dashboard_data(DEFAULT_DATASET_PATH)
    DEFAULT_DATA_ERROR = None
except Exception as exc:
    DEFAULT_DF = pd.DataFrame(columns=REQUIRED_COLUMNS + ["SEMANA"])
    DEFAULT_SHEET = None
    DEFAULT_DATA_ERROR = str(exc)

DEFAULT_DATA_JSON = serialize_df(DEFAULT_DF) if not DEFAULT_DF.empty else None
if DEFAULT_DF.empty:
    DEFAULT_META = make_dataset_meta(
        status="error",
        source="default",
        message=DEFAULT_DATA_ERROR or "No default dataset loaded. Upload an Excel file to begin.",
        sheet=None,
        rows=0,
        filename=str(DEFAULT_DATASET_PATH),
    )
else:
    DEFAULT_META = make_dataset_meta(
        status="ok",
        source="default",
        message=f"Loaded local dataset: {DEFAULT_DATASET_PATH.name} (sheet {DEFAULT_SHEET})",
        sheet=DEFAULT_SHEET,
        rows=len(DEFAULT_DF),
        filename=DEFAULT_DATASET_PATH.name,
    )

MODEL_METRICS = load_saved_metrics()

min_date, max_date, leather_type_options, family_options, mlp_leather_type_options = compute_filter_options(DEFAULT_DF)


def kpi_card(label: str, value_id: str) -> dmc.Paper:
    return dmc.Paper(
        className="kpi-card",
        withBorder=True,
        radius="lg",
        p="md",
        children=[
            dmc.Text(label, c="dimmed", fz="xs", tt="uppercase", fw=600),
            dmc.Title(id=value_id, order=3),
        ],
    )


app = dash.Dash(__name__)
app.title = "Leather Operations MLP Dashboard"


app.layout = dmc.MantineProvider(
    theme={
        "primaryColor": "teal",
        "defaultRadius": "md",
        "fontFamily": "Space Grotesk, Inter, sans-serif",
    },
    children=html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="dataset-store", data=DEFAULT_DATA_JSON),
            dcc.Store(id="dataset-meta", data=DEFAULT_META),
            html.Div(className="orb orb-a"),
            html.Div(className="orb orb-b"),
            html.Div(className="orb orb-c"),
            dmc.Container(
                size="xl",
                className="page",
                children=[
                    dmc.Paper(
                        className="hero-panel",
                        withBorder=True,
                        radius="xl",
                        p="xl",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="flex-start",
                                children=[
                                    dmc.Stack(
                                        gap=2,
                                        children=[
                                            dmc.Badge(
                                                "Leather Tanning / Retanning Analytics",
                                                variant="gradient",
                                                gradient={"from": "teal", "to": "cyan", "deg": 45},
                                            ),
                                            dmc.Title("Operations Dashboard + MLP Forecast", order=1),
                                            dmc.Text(
                                                "Upload your own Excel data, explore KPIs, and train a prediction model locally.",
                                                c="dimmed",
                                            ),
                                        ],
                                    ),
                                    dmc.Group(
                                        gap="xs",
                                        children=[
                                            dmc.Anchor("GitHub", href=REPO_URL, target="_blank"),
                                            html.Div(id="mlp-model-badge-slot", children=make_mlp_model_badge(MODEL_METRICS)),
                                        ],
                                    ),
                                ],
                            ),
                            dmc.Space(h="md"),
                            dmc.Group(
                                gap="sm",
                                children=[
                                    dmc.Badge(
                                        "Dataset ready" if not DEFAULT_DF.empty else "Dataset not loaded",
                                        id="dataset-ready-badge",
                                        color="teal" if not DEFAULT_DF.empty else "red",
                                        variant="light",
                                    ),
                                    dmc.Badge(
                                        f"Sheet: {DEFAULT_SHEET}" if DEFAULT_SHEET else "Sheet: n/a",
                                        id="sheet-badge",
                                        color="gray",
                                        variant="outline",
                                    ),
                                    dmc.Badge(
                                        f"Rows: {len(DEFAULT_DF):,}" if not DEFAULT_DF.empty else "Rows: 0",
                                        id="rows-badge",
                                        color="gray",
                                        variant="outline",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.Space(h="md"),
                    dmc.Alert(
                        DEFAULT_META.get("message", "Upload a dataset to start."),
                        id="data-source-alert",
                        color="red" if DEFAULT_META.get("status") == "error" else "teal",
                        variant="light",
                    ),
                    dmc.Space(h="md"),
                    dmc.SimpleGrid(
                        cols={"base": 1, "sm": 2, "md": 3, "lg": 5},
                        spacing="md",
                        children=[
                            kpi_card("Top Family", "card-familia"),
                            kpi_card("Top Yield", "card-rendimiento"),
                            kpi_card("Top Leather Type", "card-cuero"),
                            kpi_card("Total Area", "card-area"),
                            kpi_card("Total Pieces", "card-pzs"),
                        ],
                    ),
                    dmc.Space(h="md"),
                    dmc.Grid(
                        gutter="md",
                        children=[
                            dmc.GridCol(
                                span={"base": 12, "md": 3},
                                children=dmc.Paper(
                                    className="glass-card",
                                    withBorder=True,
                                    radius="lg",
                                    p="md",
                                    children=[
                                        dmc.Title("Dataset & Filters", order=4),
                                        dmc.Text(
                                            "Upload an Excel file. Processing stays local on your machine.",
                                            c="dimmed",
                                            fz="sm",
                                        ),
                                        dmc.Space(h="sm"),
                                        dcc.Upload(
                                            id="dataset-upload",
                                            multiple=False,
                                            children=dmc.Button("Upload Excel (.xlsx)", variant="outline", fullWidth=True),
                                        ),
                                        dmc.Space(h="xs"),
                                        dmc.Button(
                                            "Use bundled sample dataset",
                                            id="reset-dataset",
                                            variant="subtle",
                                            color="gray",
                                            fullWidth=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.DatePickerInput(
                                            id="date-range",
                                            label="Date range",
                                            type="range",
                                            value=[min_date.date().isoformat(), max_date.date().isoformat()],
                                            minDate=min_date.date().isoformat(),
                                            maxDate=max_date.date().isoformat(),
                                            valueFormat="YYYY-MM-DD",
                                            numberOfColumns=2,
                                            w="100%",
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.Select(
                                            id="tipo-cuero-select",
                                            label="Leather type",
                                            value="ALL",
                                            data=leather_type_options,
                                            searchable=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.NumberInput(
                                            id="min-pzs",
                                            label="Minimum pieces for family ranking",
                                            value=100,
                                            min=0,
                                        ),
                                    ],
                                ),
                            ),
                            dmc.GridCol(
                                span={"base": 12, "md": 9},
                                children=dmc.Paper(
                                    className="glass-card",
                                    withBorder=True,
                                    radius="lg",
                                    p="md",
                                    children=dcc.Graph(id="weekly-area-chart", config={"displayModeBar": False}),
                                ),
                            ),
                        ],
                    ),
                    dmc.Space(h="md"),
                    dmc.SimpleGrid(
                        cols={"base": 1, "md": 2},
                        spacing="md",
                        children=[
                            dmc.Paper(
                                className="glass-card",
                                withBorder=True,
                                radius="lg",
                                p="md",
                                children=dcc.Graph(id="familia-chart", config={"displayModeBar": False}),
                            ),
                            dmc.Paper(
                                className="glass-card",
                                withBorder=True,
                                radius="lg",
                                p="md",
                                children=dcc.Graph(id="cuero-chart", config={"displayModeBar": False}),
                            ),
                        ],
                    ),
                    dmc.Space(h="md"),
                    dmc.Paper(
                        className="glass-card",
                        withBorder=True,
                        radius="lg",
                        p="md",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="flex-end",
                                children=[
                                    dmc.Stack(
                                        gap=2,
                                        children=[
                                            dmc.Title("Weekly Yield Mix Explainer (Prototype)", order=4),
                                            dmc.Text(
                                                "Shows whether yield changed mainly due to family mix (what was processed) or execution (how it was processed).",
                                                c="dimmed",
                                                fz="sm",
                                            ),
                                        ],
                                    ),
                                    dmc.Select(
                                        id="mix-week-select",
                                        label="Week to explain",
                                        w=220,
                                        data=[],
                                        placeholder="Select week",
                                        clearable=False,
                                    ),
                                ],
                            ),
                            dmc.Space(h="sm"),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 2, "md": 3, "lg": 6},
                                spacing="md",
                                children=[
                                    kpi_card("Previous Yield", "mix-kpi-prev"),
                                    kpi_card("Current Yield", "mix-kpi-current"),
                                    kpi_card("Delta Yield", "mix-kpi-delta"),
                                    kpi_card("Mix Effect", "mix-kpi-mix"),
                                    kpi_card("Execution Effect", "mix-kpi-exec"),
                                    kpi_card("Interaction", "mix-kpi-interaction"),
                                ],
                            ),
                            dmc.Space(h="md"),
                            dmc.SimpleGrid(
                                cols={"base": 1, "lg": 2},
                                spacing="md",
                                children=[
                                    dcc.Graph(id="mix-waterfall", config={"displayModeBar": False}),
                                    dcc.Graph(id="mix-share-chart", config={"displayModeBar": False}),
                                ],
                            ),
                            dmc.Space(h="md"),
                            dcc.Graph(id="mix-driver-chart", config={"displayModeBar": False}),
                            dmc.Space(h="xs"),
                            dmc.Alert(
                                id="mix-narrative",
                                color="blue",
                                variant="light",
                                children="Upload data and select a week to explain yield movement.",
                            ),
                        ],
                    ),
                    dmc.Space(h="md"),
                    dmc.Grid(
                        gutter="md",
                        children=[
                            dmc.GridCol(
                                span={"base": 12, "md": 4},
                                children=dmc.Paper(
                                    className="glass-card",
                                    withBorder=True,
                                    radius="lg",
                                    p="md",
                                    children=[
                                        dmc.Title("MLP Inference", order=4),
                                        dmc.Text(
                                            "Forecast total area (ft2) using family, leather type, and piece count.",
                                            c="dimmed",
                                            fz="sm",
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.Select(
                                            id="mlp-familia",
                                            label="Family",
                                            value=family_options[0],
                                            data=family_options,
                                            searchable=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.Select(
                                            id="mlp-tipo-cuero",
                                            label="Leather type",
                                            value=mlp_leather_type_options[0],
                                            data=mlp_leather_type_options,
                                            searchable=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.NumberInput(id="mlp-pzs", label="Pieces (PZS)", value=220, min=1),
                                        dmc.Space(h="sm"),
                                        dmc.Button("Run prediction", id="mlp-button", fullWidth=True),
                                        dmc.Space(h="xs"),
                                        dmc.Button(
                                            "Train model from current dataset",
                                            id="mlp-train-button",
                                            fullWidth=True,
                                            color="indigo",
                                            variant="light",
                                        ),
                                        dmc.Space(h="xs"),
                                        dmc.Stack(
                                            id="mlp-train-status",
                                            gap="xs",
                                            children=[
                                                dmc.Text(
                                                    "Train once after uploading data to refresh the model.",
                                                    c="dimmed",
                                                    fz="sm",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                            dmc.GridCol(
                                span={"base": 12, "md": 8},
                                children=dmc.Paper(
                                    className="glass-card",
                                    withBorder=True,
                                    radius="lg",
                                    p="md",
                                    children=[
                                        dmc.Title("Prediction Output", order=4),
                                        dmc.Space(h="sm"),
                                        dmc.Stack(
                                            [dmc.Text("Click Run prediction to evaluate the model.")],
                                            id="mlp-result",
                                            gap="xs",
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ),
)


@app.callback(
    Output("dataset-store", "data"),
    Output("dataset-meta", "data"),
    Input("dataset-upload", "contents"),
    Input("reset-dataset", "n_clicks"),
    State("dataset-upload", "filename"),
    State("dataset-store", "data"),
    prevent_initial_call=True,
)
def handle_dataset_upload(contents, _reset_clicks, filename, current_data):
    trigger = dash.ctx.triggered_id

    if trigger == "reset-dataset":
        return DEFAULT_DATA_JSON, DEFAULT_META

    if not contents:
        raise PreventUpdate

    try:
        _, payload = contents.split(",", 1)
        file_bytes = base64.b64decode(payload)
        uploaded_df, selected_sheet = load_dashboard_data_from_bytes(file_bytes)
        data_json = serialize_df(uploaded_df)
        meta = make_dataset_meta(
            status="ok",
            source="upload",
            message=f"Loaded uploaded dataset: {filename or 'uploaded.xlsx'} (sheet {selected_sheet})",
            sheet=selected_sheet,
            rows=len(uploaded_df),
            filename=filename,
        )
        return data_json, meta
    except Exception as exc:
        meta = make_dataset_meta(
            status="error",
            source="upload",
            message=f"Upload failed: {exc}",
            sheet=None,
            rows=0,
            filename=filename,
        )
        return current_data, meta


@app.callback(
    Output("data-source-alert", "children"),
    Output("data-source-alert", "color"),
    Output("dataset-ready-badge", "children"),
    Output("dataset-ready-badge", "color"),
    Output("sheet-badge", "children"),
    Output("rows-badge", "children"),
    Output("date-range", "value"),
    Output("date-range", "minDate"),
    Output("date-range", "maxDate"),
    Output("tipo-cuero-select", "data"),
    Output("tipo-cuero-select", "value"),
    Output("mlp-familia", "data"),
    Output("mlp-familia", "value"),
    Output("mlp-tipo-cuero", "data"),
    Output("mlp-tipo-cuero", "value"),
    Output("mix-week-select", "data"),
    Output("mix-week-select", "value"),
    Input("dataset-store", "data"),
    Input("dataset-meta", "data"),
)
def sync_ui_to_dataset(dataset_json, dataset_meta):
    df = deserialize_df(dataset_json)
    meta = dataset_meta or {}

    min_dt, max_dt, leather_opts, family_opts, mlp_leather_opts = compute_filter_options(df)
    week_opts, week_value = compute_week_options(df)
    date_value = [min_dt.date().isoformat(), max_dt.date().isoformat()]

    alert_message = meta.get("message") or "Upload a dataset to start."
    alert_color = "red" if meta.get("status") == "error" else "teal"

    dataset_ready = not df.empty
    ready_text = "Dataset ready" if dataset_ready else "Dataset not loaded"
    ready_color = "teal" if dataset_ready else "red"

    sheet_text = f"Sheet: {meta.get('sheet')}" if meta.get("sheet") else "Sheet: n/a"
    rows_text = f"Rows: {len(df):,}" if dataset_ready else "Rows: 0"

    leather_value = "ALL" if "ALL" in leather_opts else leather_opts[0]
    family_value = family_opts[0] if family_opts else "NO DATA"
    mlp_leather_value = mlp_leather_opts[0] if mlp_leather_opts else "NO DATA"

    return (
        alert_message,
        alert_color,
        ready_text,
        ready_color,
        sheet_text,
        rows_text,
        date_value,
        min_dt.date().isoformat(),
        max_dt.date().isoformat(),
        leather_opts,
        leather_value,
        family_opts,
        family_value,
        mlp_leather_opts,
        mlp_leather_value,
        week_opts,
        week_value,
    )


@app.callback(
    Output("card-familia", "children"),
    Output("card-rendimiento", "children"),
    Output("card-cuero", "children"),
    Output("card-area", "children"),
    Output("card-pzs", "children"),
    Output("weekly-area-chart", "figure"),
    Output("familia-chart", "figure"),
    Output("cuero-chart", "figure"),
    Input("dataset-store", "data"),
    Input("date-range", "value"),
    Input("tipo-cuero-select", "value"),
    Input("min-pzs", "value"),
)
def update_dashboard(dataset_json, date_range, selected_tipo, min_pzs):
    df = deserialize_df(dataset_json)

    if df.empty:
        empty_fig = empty_chart("No data", "Load a valid dataset to render charts.")
        return ("N/A", "N/A", "N/A", "0 ft2", "0", empty_fig, empty_fig, empty_fig)

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        start_date = df["FECHA"].min()
        end_date = df["FECHA"].max()

    if end_date < start_date:
        start_date, end_date = end_date, start_date

    selected_tipo = selected_tipo or "ALL"
    min_pzs = 0 if min_pzs is None else min_pzs

    filtered = df[(df["FECHA"] >= start_date) & (df["FECHA"] <= end_date)].copy()
    if selected_tipo != "ALL":
        filtered = filtered[filtered["TIPO DE CUERO"] == selected_tipo].copy()

    if filtered.empty:
        empty_fig = empty_chart("No data for this filter", "Try a wider date range or another leather type.")
        return ("N/A", "No data", "N/A", "0 ft2", "0", empty_fig, empty_fig, empty_fig)

    total_area = filtered["AREA TOTAL (ft2)"].sum()
    total_pzs = filtered["PZS"].sum()

    weekly = filtered.groupby("SEMANA", as_index=False)["AREA TOTAL (ft2)"].sum().sort_values("SEMANA")
    weekly_fig = go.Figure(
        data=[
            go.Scatter(
                x=weekly["SEMANA"],
                y=weekly["AREA TOTAL (ft2)"],
                mode="lines+markers",
                line=dict(color="#0ea5a4", width=3),
                marker=dict(size=6, color="#f59e0b"),
                fill="tozeroy",
                fillcolor="rgba(14,165,164,0.18)",
                name="Weekly area",
            )
        ]
    )
    style_chart(weekly_fig, "Weekly Total Area")
    weekly_fig.update_xaxes(title="Week")
    weekly_fig.update_yaxes(title="Area (ft2)")

    family_summary = (
        filtered.groupby("FAMILIA", as_index=False)
        .agg({"AREA TOTAL (ft2)": "sum", "PZS": "sum"})
    )
    family_summary = family_summary[family_summary["PZS"] >= min_pzs].copy()
    family_summary["RENDIMIENTO"] = family_summary["AREA TOTAL (ft2)"] / family_summary["PZS"]
    family_summary = family_summary.replace([float("inf"), -float("inf")], 0).fillna(0)
    family_summary = family_summary.sort_values("RENDIMIENTO", ascending=False)

    if family_summary.empty:
        top_family = "N/A"
        top_yield = 0.0
        family_fig = empty_chart("Top Families by Yield", "No family meets the minimum pieces threshold.")
    else:
        top_family = str(family_summary.iloc[0]["FAMILIA"])
        top_yield = float(family_summary.iloc[0]["RENDIMIENTO"])
        top10 = family_summary.head(10)
        family_fig = go.Figure(
            data=[
                go.Bar(
                    x=top10["FAMILIA"],
                    y=top10["RENDIMIENTO"],
                    marker=dict(color="#06b6d4", line=dict(width=1, color="rgba(15,23,42,0.2)")),
                    name="Yield",
                )
            ]
        )
        style_chart(family_fig, "Top 10 Families by Yield")
        family_fig.update_xaxes(title="Family")
        family_fig.update_yaxes(title="Yield (ft2 per piece)")

    leather_summary = (
        filtered.groupby("TIPO DE CUERO", as_index=False)["AREA TOTAL (ft2)"]
        .sum()
        .sort_values("AREA TOTAL (ft2)", ascending=False)
    )

    if leather_summary.empty:
        top_leather = "N/A"
        leather_fig = empty_chart("Top Leather Types", "No leather-type rows available.")
    else:
        top_leather = str(leather_summary.iloc[0]["TIPO DE CUERO"])
        leather_top10 = leather_summary.head(10)
        leather_fig = go.Figure(
            data=[
                go.Bar(
                    x=leather_top10["AREA TOTAL (ft2)"],
                    y=leather_top10["TIPO DE CUERO"],
                    orientation="h",
                    marker=dict(color="#3b82f6", line=dict(width=1, color="rgba(15,23,42,0.2)")),
                    name="Area",
                )
            ]
        )
        style_chart(leather_fig, "Top Leather Types by Total Area")
        leather_fig.update_xaxes(title="Area (ft2)")
        leather_fig.update_yaxes(title="Leather type", categoryorder="total ascending")

    return (
        top_family,
        f"{top_yield:,.2f} ft2/piece",
        top_leather,
        f"{total_area:,.0f} ft2",
        f"{total_pzs:,.0f}",
        weekly_fig,
        family_fig,
        leather_fig,
    )


@app.callback(
    Output("mix-kpi-prev", "children"),
    Output("mix-kpi-current", "children"),
    Output("mix-kpi-delta", "children"),
    Output("mix-kpi-mix", "children"),
    Output("mix-kpi-exec", "children"),
    Output("mix-kpi-interaction", "children"),
    Output("mix-waterfall", "figure"),
    Output("mix-share-chart", "figure"),
    Output("mix-driver-chart", "figure"),
    Output("mix-narrative", "children"),
    Output("mix-narrative", "color"),
    Input("dataset-store", "data"),
    Input("date-range", "value"),
    Input("tipo-cuero-select", "value"),
    Input("min-pzs", "value"),
    Input("mix-week-select", "value"),
)
def update_mix_explainer(dataset_json, date_range, selected_tipo, min_pzs, mix_week):
    df = deserialize_df(dataset_json)

    fallback_fig = empty_chart("Yield Mix Explainer", "Need at least two weeks of data for this analysis.")
    if df.empty:
        return (
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fallback_fig,
            fallback_fig,
            fallback_fig,
            "Carga un dataset para activar esta sección.",
            "gray",
        )

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        start_date = df["FECHA"].min()
        end_date = df["FECHA"].max()

    if end_date < start_date:
        start_date, end_date = end_date, start_date

    selected_tipo = selected_tipo or "ALL"
    filtered = df[(df["FECHA"] >= start_date) & (df["FECHA"] <= end_date)].copy()
    if selected_tipo != "ALL":
        filtered = filtered[filtered["TIPO DE CUERO"] == selected_tipo].copy()

    if filtered.empty:
        return (
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fallback_fig,
            fallback_fig,
            fallback_fig,
            "No hay datos para ese filtro de fecha/cuero.",
            "yellow",
        )

    family_week = (
        filtered.groupby(["SEMANA", "FAMILIA"], as_index=False)
        .agg(area=("AREA TOTAL (ft2)", "sum"), pzs=("PZS", "sum"))
    )
    family_week = family_week[family_week["pzs"] > 0].copy()
    family_week["yield"] = family_week["area"] / family_week["pzs"]

    threshold = max(float(min_pzs or 0), 0)
    if threshold > 0:
        threshold_slice = family_week[family_week["pzs"] >= threshold].copy()
        if not threshold_slice.empty:
            family_week = threshold_slice

    week_values = sorted(pd.to_datetime(family_week["SEMANA"].dropna().unique()))
    if len(week_values) < 2:
        return (
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fallback_fig,
            fallback_fig,
            fallback_fig,
            "Necesitas al menos dos semanas con familias válidas para comparar rendimiento.",
            "yellow",
        )

    focus_week = pd.to_datetime(mix_week) if mix_week else week_values[-1]
    focus_week = pd.Timestamp(focus_week).normalize()
    week_map = {pd.Timestamp(w).normalize(): pd.Timestamp(w) for w in week_values}
    if focus_week not in week_map:
        focus_week = pd.Timestamp(week_values[-1]).normalize()

    ordered_norm = [pd.Timestamp(w).normalize() for w in week_values]
    focus_idx = ordered_norm.index(focus_week)
    if focus_idx == 0:
        return (
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fallback_fig,
            fallback_fig,
            fallback_fig,
            "Selecciona una semana que tenga semana previa dentro del filtro.",
            "yellow",
        )

    curr_week = pd.Timestamp(week_values[focus_idx])
    prev_week = pd.Timestamp(week_values[focus_idx - 1])

    curr_rows = family_week[family_week["SEMANA"] == curr_week][["FAMILIA", "area", "pzs", "yield"]].rename(
        columns={"area": "area_curr", "pzs": "pzs_curr", "yield": "yield_curr"}
    )
    prev_rows = family_week[family_week["SEMANA"] == prev_week][["FAMILIA", "area", "pzs", "yield"]].rename(
        columns={"area": "area_prev", "pzs": "pzs_prev", "yield": "yield_prev"}
    )

    merged = prev_rows.merge(curr_rows, on="FAMILIA", how="outer").fillna(0)

    prev_total_pzs = float(merged["pzs_prev"].sum())
    curr_total_pzs = float(merged["pzs_curr"].sum())
    prev_total_area = float(merged["area_prev"].sum())
    curr_total_area = float(merged["area_curr"].sum())

    if prev_total_pzs <= 0 or curr_total_pzs <= 0:
        return (
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            fallback_fig,
            fallback_fig,
            fallback_fig,
            "Una de las semanas tiene piezas en cero; no se puede descomponer rendimiento.",
            "yellow",
        )

    merged["share_prev"] = merged["pzs_prev"] / prev_total_pzs
    merged["share_curr"] = merged["pzs_curr"] / curr_total_pzs
    merged["share_delta"] = merged["share_curr"] - merged["share_prev"]
    merged["share_delta_pct"] = merged["share_delta"] * 100
    merged["mix_contrib"] = merged["share_delta"] * merged["yield_prev"]
    merged["exec_contrib"] = merged["share_prev"] * (merged["yield_curr"] - merged["yield_prev"])
    merged["interaction_contrib"] = merged["share_delta"] * (merged["yield_curr"] - merged["yield_prev"])

    previous_yield = prev_total_area / prev_total_pzs
    current_yield = curr_total_area / curr_total_pzs
    delta_yield = current_yield - previous_yield
    mix_effect = float(merged["mix_contrib"].sum())
    execution_effect = float(merged["exec_contrib"].sum())
    interaction_effect = float(merged["interaction_contrib"].sum())

    curr_label = curr_week.strftime("%Y-%m-%d")
    prev_label = prev_week.strftime("%Y-%m-%d")

    waterfall_fig = make_mix_waterfall_figure(
        previous_yield=previous_yield,
        mix_effect=mix_effect,
        execution_effect=execution_effect,
        interaction_effect=interaction_effect,
        current_yield=current_yield,
    )
    share_fig = make_mix_share_figure(merged=merged, prev_label=prev_label, curr_label=curr_label)
    driver_fig = make_mix_driver_scatter(merged=merged, previous_yield=previous_yield)

    negative_mix = merged.sort_values("mix_contrib").head(3)
    negative_mix = negative_mix[negative_mix["mix_contrib"] < 0]
    if negative_mix.empty:
        drivers = "No major negative family-mix drivers this week."
    else:
        drivers = ", ".join(
            f"{row.FAMILIA} ({row.mix_contrib:.3f})"
            for row in negative_mix.itertuples()
        )

    if mix_effect < 0 and abs(mix_effect) > abs(execution_effect):
        headline = (
            f"Semana {curr_label} vs {prev_label}: el rendimiento bajó principalmente por mezcla de familias "
            "(entraron más familias de bajo rendimiento)."
        )
        color = "red"
    elif execution_effect < 0 and abs(execution_effect) > abs(mix_effect):
        headline = (
            f"Semana {curr_label} vs {prev_label}: la caída se explica más por ejecución dentro de familias "
            "que por mezcla."
        )
        color = "orange"
    else:
        headline = (
            f"Semana {curr_label} vs {prev_label}: el efecto es mixto entre composición de familias y ejecución."
        )
        color = "blue"

    narrative = (
        f"{headline} Drivers negativos de mezcla: {drivers}. "
        f"ΔYield={delta_yield:+.3f}, Mix={mix_effect:+.3f}, Execution={execution_effect:+.3f}, Interaction={interaction_effect:+.3f}."
    )

    return (
        f"{previous_yield:,.3f} ft2/piece",
        f"{current_yield:,.3f} ft2/piece",
        f"{delta_yield:+,.3f}",
        f"{mix_effect:+,.3f}",
        f"{execution_effect:+,.3f}",
        f"{interaction_effect:+,.3f}",
        waterfall_fig,
        share_fig,
        driver_fig,
        narrative,
        color,
    )


@app.callback(
    Output("mlp-train-status", "children"),
    Output("mlp-model-badge-slot", "children"),
    Input("mlp-train-button", "n_clicks"),
    State("dataset-store", "data"),
    State("dataset-meta", "data"),
    prevent_initial_call=True,
)
def train_model_from_current_dataset(n_clicks, dataset_json, dataset_meta):
    if n_clicks is None:
        raise PreventUpdate

    if train_and_save_from_dataframe is None:
        return [
            dmc.Text("Could not import model-training backend.", c="red", fw=700),
            dmc.Text(MLP_IMPORT_ERROR or "Unknown import error", c="dimmed", fz="sm"),
        ], make_mlp_model_badge(load_saved_metrics())

    df = deserialize_df(dataset_json)
    if df.empty:
        return [dmc.Text("No dataset loaded. Upload an Excel file first.", c="red", fw=700)], make_mlp_model_badge(None)

    source_label = (dataset_meta or {}).get("sheet") or (dataset_meta or {}).get("filename") or "uploaded-dataset"

    try:
        metrics = train_and_save_from_dataframe(df, source_label=source_label)
    except Exception as exc:
        return [dmc.Text(f"Training failed: {exc}", c="red", fw=700)], make_mlp_model_badge(load_saved_metrics())

    status_children = [
        dmc.Badge("Training complete", color="teal", variant="light"),
        dmc.Text(f"Rows used: {metrics.get('rows_used', 0):,}"),
        dmc.Text(f"R2: {metrics.get('r2', 0):.3f} | MAE: {metrics.get('mae', 0):,.2f} | RMSE: {metrics.get('rmse', 0):,.2f}"),
    ]
    return status_children, make_mlp_model_badge(metrics)


@app.callback(
    Output("mlp-result", "children"),
    Input("mlp-button", "n_clicks"),
    State("mlp-familia", "value"),
    State("mlp-tipo-cuero", "value"),
    State("mlp-pzs", "value"),
)
def update_mlp_prediction(n_clicks, familia, tipo_cuero, pzs):
    if n_clicks is None:
        return [dmc.Text("Click Run prediction to evaluate the trained model.")]

    if predict_area_total is None:
        return [
            dmc.Text("Could not import model inference backend.", c="red", fw=700),
            dmc.Text(MLP_IMPORT_ERROR or "Unknown import error", c="dimmed", fz="sm"),
        ]

    if not familia or not tipo_cuero or pzs is None:
        return [dmc.Text("Please select family, leather type, and piece count.", c="red", fw=700)]

    if float(pzs) <= 0:
        return [dmc.Text("Piece count must be greater than 0.", c="red", fw=700)]

    try:
        result = predict_area_total(familia=familia, tipo_cuero=tipo_cuero, pzs=float(pzs))
    except Exception as exc:
        return [dmc.Text(f"Model error: {exc}", c="red", fw=700)]

    predicted_area = result["predicted_area"]
    predicted_yield = result["predicted_rendimiento"]
    metrics = result.get("metrics", {})

    return [
        dmc.Badge("Prediction complete", color="teal", variant="light"),
        dmc.Text(f"Family: {familia}", fw=700),
        dmc.Text(f"Leather type: {tipo_cuero}"),
        dmc.Text(f"Pieces: {float(pzs):,.0f}"),
        dmc.Space(h="xs"),
        dmc.Title(f"{predicted_area:,.2f} ft2", order=2, c="teal"),
        dmc.Text(f"Predicted yield: {predicted_yield:,.2f} ft2/piece", fw=600),
        dmc.Space(h="xs"),
        dmc.Divider(),
        dmc.Text("Model quality snapshot", fz="sm", c="dimmed", fw=600),
        dmc.Group(
            gap="md",
            children=[
                dmc.Badge(f"MAE {metrics.get('mae', 0):,.2f}", color="gray", variant="outline"),
                dmc.Badge(f"RMSE {metrics.get('rmse', 0):,.2f}", color="gray", variant="outline"),
                dmc.Badge(f"R2 {metrics.get('r2', 0):.3f}", color="gray", variant="outline"),
                dmc.Badge(f"Rows {metrics.get('rows_used', 0):,}", color="gray", variant="outline"),
            ],
        ),
    ]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
