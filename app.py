from __future__ import annotations

from pathlib import Path
import os

import dash
from dash import Input, Output, State, dcc, html
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from mlp_recurtido import predict_area_total
    MLP_IMPORT_ERROR = None
except Exception as exc:
    predict_area_total = None
    MLP_IMPORT_ERROR = str(exc)


BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = Path(os.getenv("RECURTIDO_EXCEL_PATH", BASE_DIR / "data" / "datosProd.xlsx"))
PREFERRED_SHEET = os.getenv("RECURTIDO_SHEET_NAME", "RECURTIDO")
REPO_URL = "https://github.com/js0596a/recurtido-mlp-dashboard"
REQUIRED_COLUMNS = ["FECHA", "TIPO DE CUERO", "FAMILIA", "PZS", "AREA TOTAL (ft2)"]
METRICS_PATH = BASE_DIR / "artifacts" / "recurtido_metrics.joblib"


def fix_excel_date(column: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(column):
        return column

    numeric_dates = pd.to_numeric(column, errors="coerce")
    if numeric_dates.notna().mean() > 0.8:
        return pd.to_datetime(numeric_dates, unit="D", origin="1899-12-30", errors="coerce")

    return pd.to_datetime(column, errors="coerce")


def select_sheet_with_columns(excel_path: Path, preferred_sheet: str) -> tuple[pd.DataFrame, str]:
    try:
        workbook = pd.ExcelFile(excel_path)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    sheet_names = workbook.sheet_names
    candidates = [preferred_sheet] + [name for name in sheet_names if name != preferred_sheet]

    for sheet in candidates:
        if sheet not in sheet_names:
            continue
        candidate_df = pd.read_excel(excel_path, sheet_name=sheet)
        candidate_df.columns = candidate_df.columns.str.strip()
        if all(col in candidate_df.columns for col in REQUIRED_COLUMNS):
            return candidate_df, sheet

    raise ValueError(
        "No worksheet contains required columns "
        f"{REQUIRED_COLUMNS}. Available sheets: {sheet_names}"
    )


def load_dashboard_data(excel_path: Path) -> tuple[pd.DataFrame, str]:
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found at {excel_path}. "
            "Set RECURTIDO_EXCEL_PATH to your local dataset path."
        )

    raw_df, selected_sheet = select_sheet_with_columns(excel_path, PREFERRED_SHEET)

    df = raw_df[REQUIRED_COLUMNS].copy()
    df["FECHA"] = fix_excel_date(df["FECHA"])

    df["TIPO DE CUERO"] = df["TIPO DE CUERO"].astype(str).str.strip().str.upper()
    df["FAMILIA"] = df["FAMILIA"].astype(str).str.strip().str.upper()

    df["PZS"] = pd.to_numeric(df["PZS"], errors="coerce")
    df["AREA TOTAL (ft2)"] = pd.to_numeric(df["AREA TOTAL (ft2)"], errors="coerce")

    df = df.dropna(subset=["FECHA", "PZS", "AREA TOTAL (ft2)"])

    df["TIPO DE CUERO"] = df["TIPO DE CUERO"].replace({"NAN": "SIN DATO", "-": "SIN DATO", "": "SIN DATO"})
    df["FAMILIA"] = df["FAMILIA"].replace({"NAN": "SIN DATO", "-": "SIN DATO", "": "SIN DATO"})

    df = df[(df["PZS"] > 0) & (df["AREA TOTAL (ft2)"] > 0)].copy()
    df["SEMANA"] = df["FECHA"].dt.to_period("W").dt.start_time

    return df, selected_sheet


def load_saved_metrics() -> dict | None:
    if joblib is None:
        return None
    if not METRICS_PATH.exists():
        return None
    try:
        metrics = joblib.load(METRICS_PATH)
        return metrics if isinstance(metrics, dict) else None
    except Exception:
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


try:
    DF, SELECTED_SHEET = load_dashboard_data(EXCEL_PATH)
    DATA_LOAD_ERROR = None
except Exception as exc:
    DF = pd.DataFrame(columns=REQUIRED_COLUMNS + ["SEMANA"])
    SELECTED_SHEET = None
    DATA_LOAD_ERROR = str(exc)


MODEL_METRICS = load_saved_metrics()

if DF.empty:
    min_date = pd.Timestamp.today().normalize()
    max_date = pd.Timestamp.today().normalize()
    tipo_cuero_options = ["ALL"]
    familia_options = ["NO DATA"]
    mlp_tipo_cuero_options = ["NO DATA"]
else:
    min_date = DF["FECHA"].min()
    max_date = DF["FECHA"].max()
    tipo_cuero_options = ["ALL"] + sorted(DF["TIPO DE CUERO"].dropna().unique())
    familia_options = sorted(DF["FAMILIA"].dropna().unique())
    mlp_tipo_cuero_options = sorted(DF["TIPO DE CUERO"].dropna().unique())


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
app.title = "Curtido/Recurtido MLP Dashboard"


app.layout = dmc.MantineProvider(
    theme={
        "primaryColor": "teal",
        "defaultRadius": "md",
        "fontFamily": "Space Grotesk, Inter, sans-serif",
    },
    children=html.Div(
        className="app-shell",
        children=[
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
                                                "Curtido / Recurtido Operations",
                                                variant="gradient",
                                                gradient={"from": "teal", "to": "cyan", "deg": 45},
                                            ),
                                            dmc.Title("Recurtido Analytics + MLP Forecast", order=1),
                                            dmc.Text(
                                                "Production dashboard with model-based area prediction from family, leather type, and piece count.",
                                                c="dimmed",
                                            ),
                                        ],
                                    ),
                                    dmc.Group(
                                        gap="xs",
                                        children=[
                                            dmc.Anchor("GitHub", href=REPO_URL, target="_blank"),
                                            make_mlp_model_badge(MODEL_METRICS),
                                        ],
                                    ),
                                ],
                            ),
                            dmc.Space(h="md"),
                            dmc.Group(
                                gap="sm",
                                children=[
                                    dmc.Badge(
                                        "Dataset ready" if not DATA_LOAD_ERROR else "Dataset not loaded",
                                        color="teal" if not DATA_LOAD_ERROR else "red",
                                        variant="light",
                                    ),
                                    dmc.Badge(
                                        f"Sheet: {SELECTED_SHEET}" if SELECTED_SHEET else "Sheet: n/a",
                                        color="gray",
                                        variant="outline",
                                    ),
                                    dmc.Badge(
                                        f"Rows: {len(DF):,}" if not DF.empty else "Rows: 0",
                                        color="gray",
                                        variant="outline",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.Space(h="md"),
                    dmc.Alert(
                        DATA_LOAD_ERROR
                        if DATA_LOAD_ERROR
                        else f"Loaded data from {EXCEL_PATH}",
                        color="red" if DATA_LOAD_ERROR else "teal",
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
                            kpi_card("Total PZS", "card-pzs"),
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
                                        dmc.Title("Filters", order=4),
                                        dmc.Text("Control what you see in charts and KPIs.", c="dimmed", fz="sm"),
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
                                            data=tipo_cuero_options,
                                            searchable=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.NumberInput(
                                            id="min-pzs",
                                            label="Minimum PZS for family ranking",
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
                                            "Forecast AREA TOTAL (ft2) using FAMILIA, TIPO DE CUERO, and PZS.",
                                            c="dimmed",
                                            fz="sm",
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.Select(
                                            id="mlp-familia",
                                            label="Family",
                                            value=familia_options[0],
                                            data=familia_options,
                                            searchable=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.Select(
                                            id="mlp-tipo-cuero",
                                            label="Leather type",
                                            value=mlp_tipo_cuero_options[0],
                                            data=mlp_tipo_cuero_options,
                                            searchable=True,
                                        ),
                                        dmc.Space(h="sm"),
                                        dmc.NumberInput(id="mlp-pzs", label="PZS", value=220, min=1),
                                        dmc.Space(h="sm"),
                                        dmc.Button("Run prediction", id="mlp-button", fullWidth=True),
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
    Output("card-familia", "children"),
    Output("card-rendimiento", "children"),
    Output("card-cuero", "children"),
    Output("card-area", "children"),
    Output("card-pzs", "children"),
    Output("weekly-area-chart", "figure"),
    Output("familia-chart", "figure"),
    Output("cuero-chart", "figure"),
    Input("date-range", "value"),
    Input("tipo-cuero-select", "value"),
    Input("min-pzs", "value"),
)
def update_dashboard(date_range, selected_tipo, min_pzs):
    if DF.empty:
        empty_fig = empty_chart("No data", "Load a valid dataset to render charts.")
        return ("N/A", "N/A", "N/A", "0 ft2", "0", empty_fig, empty_fig, empty_fig)

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date

    selected_tipo = selected_tipo or "ALL"
    min_pzs = 0 if min_pzs is None else min_pzs

    filtered = DF[(DF["FECHA"] >= start_date) & (DF["FECHA"] <= end_date)].copy()
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
        family_fig = empty_chart("Top Families by Yield", "No family meets the minimum PZS threshold.")
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
            dmc.Text("Could not import mlp_recurtido.py", c="red", fw=700),
            dmc.Text(MLP_IMPORT_ERROR or "Unknown import error", c="dimmed", fz="sm"),
        ]

    if not familia or not tipo_cuero or pzs is None:
        return [dmc.Text("Please select family, leather type, and PZS.", c="red", fw=700)]

    if float(pzs) <= 0:
        return [dmc.Text("PZS must be greater than 0.", c="red", fw=700)]

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
        dmc.Text(f"PZS: {float(pzs):,.0f}"),
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
