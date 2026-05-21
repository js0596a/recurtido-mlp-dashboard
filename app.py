from __future__ import annotations

from pathlib import Path
import os

import dash
from dash import Input, Output, State, dcc
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go

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
REQUIRED_DASHBOARD_COLUMNS = ["FECHA", "TIPO DE CUERO", "FAMILIA", "PZS", "AREA TOTAL (ft2)"]


def fix_excel_date(column: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(column):
        return column

    numeric_dates = pd.to_numeric(column, errors="coerce")
    if numeric_dates.notna().mean() > 0.8:
        return pd.to_datetime(numeric_dates, unit="D", origin="1899-12-30", errors="coerce")

    return pd.to_datetime(column, errors="coerce")


def load_dashboard_data(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found at {excel_path}. "
            "Set RECURTIDO_EXCEL_PATH to your local dataset path."
        )

    try:
        excel_book = pd.ExcelFile(excel_path)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    sheet_names = excel_book.sheet_names
    candidate_sheets = [PREFERRED_SHEET] + [name for name in sheet_names if name != PREFERRED_SHEET]

    selected_sheet = None
    raw_df = None
    for sheet in candidate_sheets:
        if sheet not in sheet_names:
            continue
        candidate_df = pd.read_excel(excel_path, sheet_name=sheet)
        candidate_df.columns = candidate_df.columns.str.strip()
        if all(col in candidate_df.columns for col in REQUIRED_DASHBOARD_COLUMNS):
            selected_sheet = sheet
            raw_df = candidate_df
            break

    if raw_df is None:
        raise ValueError(
            "No worksheet contains required columns "
            f"{REQUIRED_DASHBOARD_COLUMNS}. Available sheets: {sheet_names}"
        )

    missing = [col for col in REQUIRED_DASHBOARD_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing columns in selected sheet '{selected_sheet}': {missing}")

    df = raw_df[REQUIRED_DASHBOARD_COLUMNS].copy()
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

    df.attrs["selected_sheet"] = selected_sheet
    return df


def empty_chart(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#5c6670"),
    )
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def style_chart(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


try:
    DF = load_dashboard_data(EXCEL_PATH)
    SELECTED_SHEET = DF.attrs.get("selected_sheet", PREFERRED_SHEET)
    DATA_LOAD_ERROR = None
except Exception as exc:
    DF = pd.DataFrame(columns=["FECHA", "TIPO DE CUERO", "FAMILIA", "PZS", "AREA TOTAL (ft2)", "SEMANA"])
    SELECTED_SHEET = None
    DATA_LOAD_ERROR = str(exc)


if DF.empty:
    min_date = pd.Timestamp.today().normalize()
    max_date = pd.Timestamp.today().normalize()
    tipo_cuero_options = ["TODOS"]
    familia_options = ["SIN DATOS"]
    mlp_tipo_cuero_options = ["SIN DATOS"]
else:
    min_date = DF["FECHA"].min()
    max_date = DF["FECHA"].max()
    tipo_cuero_options = ["TODOS"] + sorted(DF["TIPO DE CUERO"].dropna().unique())
    familia_options = sorted(DF["FAMILIA"].dropna().unique())
    mlp_tipo_cuero_options = sorted(DF["TIPO DE CUERO"].dropna().unique())


app = dash.Dash(__name__)
app.title = "Recurtido MLP Dashboard"


app.layout = dmc.MantineProvider(
    theme={"primaryColor": "teal", "fontFamily": "Inter, sans-serif"},
    children=[
        dmc.Container(
            fluid=True,
            p="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Stack(
                            gap=0,
                            children=[
                                dmc.Title("Recurtido Analytics + MLP", order=2),
                                dmc.Text(
                                    "Production analytics dashboard with an MLP forecast workflow.",
                                    c="dimmed",
                                ),
                            ],
                        ),
                        dmc.Anchor("GitHub Repo", href=REPO_URL, target="_blank"),
                    ],
                ),
                dmc.Space(h="md"),
                dmc.Alert(
                    DATA_LOAD_ERROR
                    if DATA_LOAD_ERROR
                    else f"Loaded dataset from: {EXCEL_PATH} (sheet: {SELECTED_SHEET})",
                    color="red" if DATA_LOAD_ERROR else "teal",
                    variant="light",
                ),
                dmc.Space(h="md"),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2, "lg": 5},
                    spacing="md",
                    children=[
                        dmc.Paper([dmc.Text("Best family", c="dimmed", fz="sm"), dmc.Title(id="card-familia", order=4)], p="md", withBorder=True),
                        dmc.Paper([dmc.Text("Yield", c="dimmed", fz="sm"), dmc.Title(id="card-rendimiento", order=4)], p="md", withBorder=True),
                        dmc.Paper([dmc.Text("Top leather type", c="dimmed", fz="sm"), dmc.Title(id="card-cuero", order=4)], p="md", withBorder=True),
                        dmc.Paper([dmc.Text("Total area", c="dimmed", fz="sm"), dmc.Title(id="card-area", order=4)], p="md", withBorder=True),
                        dmc.Paper([dmc.Text("Total PZS", c="dimmed", fz="sm"), dmc.Title(id="card-pzs", order=4)], p="md", withBorder=True),
                    ],
                ),
                dmc.Space(h="md"),
                dmc.Grid(
                    children=[
                        dmc.GridCol(
                            span={"base": 12, "md": 3},
                            children=dmc.Paper(
                                withBorder=True,
                                p="md",
                                children=[
                                    dmc.Title("Filters", order=4),
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
                                        label="Tipo de cuero",
                                        value="TODOS",
                                        data=tipo_cuero_options,
                                        searchable=True,
                                    ),
                                    dmc.Space(h="sm"),
                                    dmc.NumberInput(
                                        id="min-pzs",
                                        label="Min PZS for family ranking",
                                        value=100,
                                        min=0,
                                    ),
                                ],
                            ),
                        ),
                        dmc.GridCol(
                            span={"base": 12, "md": 9},
                            children=dmc.Paper(withBorder=True, p="md", children=dcc.Graph(id="weekly-area-chart")),
                        ),
                    ]
                ),
                dmc.Space(h="md"),
                dmc.SimpleGrid(
                    cols={"base": 1, "md": 2},
                    spacing="md",
                    children=[
                        dmc.Paper(withBorder=True, p="md", children=dcc.Graph(id="familia-chart")),
                        dmc.Paper(withBorder=True, p="md", children=dcc.Graph(id="cuero-chart")),
                    ],
                ),
                dmc.Space(h="md"),
                dmc.Grid(
                    children=[
                        dmc.GridCol(
                            span={"base": 12, "md": 4},
                            children=dmc.Paper(
                                withBorder=True,
                                p="md",
                                children=[
                                    dmc.Title("MLP Forecast", order=4),
                                    dmc.Text("Predict AREA TOTAL (ft2) from family, leather type, and PZS.", c="dimmed", fz="sm"),
                                    dmc.Space(h="sm"),
                                    dmc.Select(id="mlp-familia", label="Familia", value=familia_options[0], data=familia_options, searchable=True),
                                    dmc.Space(h="sm"),
                                    dmc.Select(id="mlp-tipo-cuero", label="Tipo de cuero", value=mlp_tipo_cuero_options[0], data=mlp_tipo_cuero_options, searchable=True),
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
                                withBorder=True,
                                p="md",
                                children=[
                                    dmc.Title("Prediction Output", order=4),
                                    dmc.Space(h="sm"),
                                    dmc.Stack(
                                        [dmc.Text("Click Run prediction to evaluate the trained model.")],
                                        id="mlp-result",
                                        gap="xs",
                                    ),
                                ],
                            ),
                        ),
                    ]
                ),
            ],
        )
    ],
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
        empty_fig = empty_chart("No data", "Load private dataset to render charts.")
        return ("N/A", "N/A", "N/A", "0 ft2", "0", empty_fig, empty_fig, empty_fig)

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date

    selected_tipo = selected_tipo or "TODOS"
    min_pzs = 0 if min_pzs is None else min_pzs

    filtered = DF[(DF["FECHA"] >= start_date) & (DF["FECHA"] <= end_date)].copy()
    if selected_tipo != "TODOS":
        filtered = filtered[filtered["TIPO DE CUERO"] == selected_tipo].copy()

    if filtered.empty:
        empty_fig = empty_chart("No data for this filter", "Try a wider date range or another leather type.")
        return ("N/A", "No data", "N/A", "0 ft2", "0", empty_fig, empty_fig, empty_fig)

    total_area = filtered["AREA TOTAL (ft2)"].sum()
    total_pzs = filtered["PZS"].sum()

    weekly = (
        filtered.groupby("SEMANA")["AREA TOTAL (ft2)"]
        .sum()
        .reset_index()
        .sort_values("SEMANA")
    )
    weekly_fig = go.Figure(
        data=[
            go.Scatter(
                x=weekly["SEMANA"],
                y=weekly["AREA TOTAL (ft2)"],
                mode="lines+markers",
                line=dict(color="#0f766e", width=3),
                marker=dict(size=6),
            )
        ]
    )
    style_chart(weekly_fig, "Weekly Total Area")
    weekly_fig.update_xaxes(title="Week")
    weekly_fig.update_yaxes(title="Area (ft2)")

    familia_summary = (
        filtered.groupby("FAMILIA")
        .agg({"AREA TOTAL (ft2)": "sum", "PZS": "sum"})
        .reset_index()
    )
    familia_summary = familia_summary[familia_summary["PZS"] >= min_pzs].copy()
    familia_summary["RENDIMIENTO"] = familia_summary["AREA TOTAL (ft2)"] / familia_summary["PZS"]
    familia_summary = familia_summary.replace([float("inf"), -float("inf")], 0).fillna(0)
    familia_summary = familia_summary.sort_values("RENDIMIENTO", ascending=False)

    if familia_summary.empty:
        top_familia = "N/A"
        top_rend = 0.0
        familia_fig = empty_chart("Family yield", "No family meets minimum PZS filter.")
    else:
        top_familia = str(familia_summary.iloc[0]["FAMILIA"])
        top_rend = float(familia_summary.iloc[0]["RENDIMIENTO"])
        top_familias = familia_summary.head(10)
        familia_fig = go.Figure(
            data=[go.Bar(x=top_familias["FAMILIA"], y=top_familias["RENDIMIENTO"], marker_color="#14b8a6")]
        )
        style_chart(familia_fig, "Top 10 Families by Yield")
        familia_fig.update_xaxes(title="Family")
        familia_fig.update_yaxes(title="Yield (ft2 per piece)")

    cuero_summary = (
        filtered.groupby("TIPO DE CUERO")["AREA TOTAL (ft2)"]
        .sum()
        .reset_index()
        .sort_values("AREA TOTAL (ft2)", ascending=False)
    )
    top_cuero = str(cuero_summary.iloc[0]["TIPO DE CUERO"])
    top_cuero_area = float(cuero_summary.iloc[0]["AREA TOTAL (ft2)"])

    top_cueros = cuero_summary.head(10)
    cuero_fig = go.Figure(
        data=[go.Bar(x=top_cueros["TIPO DE CUERO"], y=top_cueros["AREA TOTAL (ft2)"], marker_color="#0284c7")]
    )
    style_chart(cuero_fig, "Top Leather Types by Area")
    cuero_fig.update_xaxes(title="Leather type")
    cuero_fig.update_yaxes(title="Area (ft2)")

    return (
        top_familia,
        f"{top_rend:,.2f} ft2/piece",
        top_cuero,
        f"{total_area:,.0f} ft2",
        f"{total_pzs:,.0f}",
        weekly_fig,
        familia_fig,
        cuero_fig,
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
            dmc.Text("Could not import MLP module.", c="red", fw=700),
            dmc.Text(MLP_IMPORT_ERROR or "Unknown error", c="dimmed", fz="sm"),
        ]

    if not familia or not tipo_cuero or pzs is None:
        return [dmc.Text("Select family, leather type, and PZS.", c="red", fw=700)]

    if float(pzs) <= 0:
        return [dmc.Text("PZS must be greater than 0.", c="red", fw=700)]

    try:
        result = predict_area_total(familia=familia, tipo_cuero=tipo_cuero, pzs=float(pzs))
    except Exception as exc:
        return [dmc.Text(f"Model error: {exc}", c="red", fw=700)]

    predicted_area = result["predicted_area"]
    predicted_rendimiento = result["predicted_rendimiento"]
    metrics = result["metrics"]

    return [
        dmc.Text(f"Familia: {familia}", fw=700),
        dmc.Text(f"Tipo de cuero: {tipo_cuero}"),
        dmc.Text(f"PZS: {float(pzs):,.0f}"),
        dmc.Space(h="sm"),
        dmc.Text(f"Predicted area: {predicted_area:,.2f} ft2", c="teal", fw=700),
        dmc.Text(f"Predicted yield: {predicted_rendimiento:,.2f} ft2/piece", c="teal", fw=700),
        dmc.Space(h="sm"),
        dmc.Text(f"MAE: {metrics['mae']:,.2f} ft2", c="dimmed", fz="sm"),
        dmc.Text(f"RMSE: {metrics['rmse']:,.2f} ft2", c="dimmed", fz="sm"),
        dmc.Text(f"R2: {metrics['r2']:.3f}", c="dimmed", fz="sm"),
        dmc.Text(f"Rows used: {metrics['rows_used']:,.0f}", c="dimmed", fz="sm"),
    ]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
