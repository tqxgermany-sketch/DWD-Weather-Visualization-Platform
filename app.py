"""
DWD Weather Data Visualization Platform — Streamlit UI
======================================================
Features:
  - Coordinate / city lookup → auto-match nearest weather station
  - Time range + parameter selection → one-click fetch
  - Single merged Plotly chart (all parameters in one view)
  - CSV export
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Ensure current directory is on sys.path for project imports
PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dwd_fetcher import (
    fetch_and_store,
    DB_PATH,
    DATE_MIN,
    DATE_MAX,
    PARAMETER_MAP,
)
from db_utils import (
    get_stations,
    get_available_parameters,
    get_date_range,
    query_daily_avg,
    export_to_csv,
)

# ─── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DWD Weather Data Platform",
    page_icon="\U0001f324",
    layout="wide",
)

# ─── Parameter metadata ─────────────────────────────────────────────────
PARAM_LABELS = {
    "temperature":   "Temperature",
    "precipitation": "Precipitation",
    "wind_speed":    "Wind Speed",
    "sunshine":      "Sunshine",
}

PARAM_UNITS = {
    "temperature":   "\u00b0C",
    "precipitation": "mm",
    "wind_speed":    "m/s",
    "sunshine":      "h",
}

PARAM_COLORS = {
    "temperature":   "#e74c3c",
    "precipitation": "#3498db",
    "wind_speed":    "#2ecc71",
    "sunshine":      "#f39c12",
}

# ─── Title ──────────────────────────────────────────────────────────────
st.title("\U0001f324 DWD German Weather Data Visualization")
st.caption(
    f"Source: Deutscher Wetterdienst (DWD) Climate Data Center  |  "
    f"Period: {DATE_MIN.date()} ~ {DATE_MAX.date()}"
)

# ══════════════════════════════════════════════════════════════════════════
# Sidebar — Configuration panel
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("\u2699 Configuration")

    # ── Location ──
    st.subheader("\U0001f4cd Location")
    location_mode = st.radio(
        "Input mode",
        ["Coordinates (lat/lon)", "German city"],
        horizontal=True,
    )

    if location_mode == "Coordinates (lat/lon)":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input(
                "Latitude",
                min_value=47.0, max_value=55.5,
                value=48.137, step=0.01, format="%.4f",
                help="Germany lat range: 47\u00b0 ~ 55.5\u00b0",
            )
        with col2:
            lon = st.number_input(
                "Longitude",
                min_value=5.5, max_value=15.5,
                value=11.576, step=0.01, format="%.4f",
                help="Germany lon range: 5.5\u00b0 ~ 15.5\u00b0",
            )
    else:
        # Preset German city coordinates
        GERMAN_CITIES = {
            "Berlin":              (52.520, 13.405),
            "Hamburg":             (53.551, 9.994),
            "Munich (M\u00fcnchen)": (48.137, 11.576),
            "Cologne (K\u00f6ln)":  (50.938, 6.960),
            "Frankfurt":           (50.111, 8.682),
            "Stuttgart":           (48.776, 9.178),
            "D\u00fcsseldorf":      (51.228, 6.773),
            "Leipzig":             (51.340, 12.375),
            "Dortmund":            (51.514, 7.465),
            "Bremen":              (53.079, 8.801),
            "Dresden":             (51.051, 13.738),
            "Hannover":            (52.376, 9.741),
            "Nuremberg (N\u00fcrnberg)": (49.452, 11.077),
        }
        city = st.selectbox("Select city", list(GERMAN_CITIES.keys()))
        lat, lon = GERMAN_CITIES[city]
        st.caption(f"Coordinates: ({lat:.4f}, {lon:.4f})")

    st.divider()

    # ── Time range ──
    st.subheader("\U0001f4c5 Time Range")
    start_date = st.date_input(
        "Start date",
        value=datetime(2024, 1, 1),
        min_value=DATE_MIN.date(),
        max_value=DATE_MAX.date(),
    )
    end_date = st.date_input(
        "End date",
        value=datetime(2024, 1, 31),
        min_value=DATE_MIN.date(),
        max_value=DATE_MAX.date(),
    )
    if start_date > end_date:
        st.error("Start date cannot be later than end date")

    st.divider()

    # ── Parameter selection ──
    st.subheader("\U0001f4ca Weather Parameters")
    params_selected = []
    for p, label in PARAM_LABELS.items():
        if st.checkbox(f"{label} ({p})", value=(p == "temperature")):
            params_selected.append(p)

    # ── Execute button ──
    fetch_btn = st.button(
        "\U0001f680 Fetch Data",
        type="primary",
        width="stretch",
        disabled=(len(params_selected) == 0 or start_date > end_date),
    )

    # ── Existing data hint ──
    existing_params = []
    try:
        stations_df = get_stations()
        if not stations_df.empty:
            for _, station_row in stations_df.iterrows():
                sid = station_row["station_id"]
                avail = get_available_parameters(sid)
                for p in params_selected:
                    date_from, date_to = get_date_range(sid, p)
                    if date_from:
                        existing_params.append(
                            (sid, station_row["name"], p, date_from, date_to)
                        )
    except Exception:
        pass

    if existing_params:
        st.divider()
        st.subheader("\U0001f4be Data already in local DB")
        seen = set()
        for sid, sname, p, dfrom, dto in existing_params:
            key = (sid, p)
            if key not in seen:
                seen.add(key)
                st.caption(
                    f"\u2022 {sname} [{sid}] \u2014 {PARAM_LABELS[p]}: {dfrom} ~ {dto}"
                )


# ══════════════════════════════════════════════════════════════════════════
# Main area — Data display
# ══════════════════════════════════════════════════════════════════════════

# ── Tab structure ──
tab_chart, tab_table, tab_export = st.tabs([
    "\U0001f4c8 Chart", "\U0001f4cb Data Table", "\U0001f4be Export"
])

# ── Data fetch logic ──
if fetch_btn:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.min.time())

    with st.spinner("Searching for nearest station and fetching data..."):
        try:
            result = fetch_and_store(
                lat=lat, lon=lon,
                parameters=params_selected,
                start_date=start_dt,
                end_date=end_dt,
            )
            st.session_state["last_result"]  = result
            st.session_state["last_params"]  = params_selected
            st.session_state["last_start"]   = start_dt
            st.session_state["last_end"]     = end_dt
            st.session_state["data_loaded"]  = True
            # Store the fetched station_id to default the selectbox later
            st.session_state["active_station"] = result["station_id"]

            st.success(
                f"\u2705 Data fetch complete!\n\n"
                f"Station: **{result['station_name']}** [{result['station_id']}]\n\n"
                f"Distance: {result['distance_km']:.1f} km  |  "
                f"Records written: {result['records_inserted']}"
            )
        except Exception as e:
            st.error(f"\u274c Fetch failed: {e}")
            st.session_state["data_loaded"] = False


# ── Load data from local DB ──
def _load_data(filter_start=None, filter_end=None):
    """Load data from the local SQLite database for the selected parameters.
    If filter_start/filter_end are provided, only data within that range is loaded.
    """
    all_data = {}
    stations_df = get_stations()
    if stations_df.empty:
        return all_data

    # Fallback: if no filter is given, use the DB's full range for each station
    query_start = datetime.combine(filter_start, datetime.min.time()) if filter_start else DATE_MIN
    query_end   = datetime.combine(filter_end,   datetime.min.time()) if filter_end   else DATE_MAX

    for _, station_row in stations_df.iterrows():
        sid = station_row["station_id"]
        for p in params_selected:
            date_from, date_to = get_date_range(sid, p)
            if date_from is None:
                continue
            df = query_daily_avg(
                sid, p,
                query_start,
                query_end,
            )
            if not df.empty:
                sid = str(sid)
                if sid not in all_data:
                    all_data[sid] = {
                        "name": station_row["name"],
                        "lat":  station_row["lat"],
                        "lon":  station_row["lon"],
                        "data": {},
                    }
                all_data[sid]["data"][p] = df
    return all_data


all_data = _load_data(filter_start=start_date, filter_end=end_date)

if not all_data:
    st.info(
        "\U0001f448 Set parameters in the sidebar and click \u2018Fetch Data\u2019, "
        "or adjust the time range to match existing data."
    )
    st.stop()

# ── Station selection (auto-matched, no manual picker) ──
station_ids = list(all_data.keys())

# Always use the last-fetched active station
selected_sid = station_ids[0]  # fallback
if "active_station" in st.session_state:
    active_sid = st.session_state["active_station"]
    if active_sid in station_ids:
        selected_sid = active_sid

station_info = all_data[selected_sid]

# Station info card
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Station name", station_info["name"])
with col2:
    st.metric("Station ID", selected_sid)
with col3:
    st.metric("Latitude", f"{station_info['lat']:.4f}\u00b0N")
with col4:
    st.metric("Longitude", f"{station_info['lon']:.4f}\u00b0E")

# ── Display parameter selector ──
available_params = [p for p in params_selected if p in station_info["data"]]
if not available_params:
    st.warning("No data available for the selected parameters. Please fetch first.")
    st.stop()

display_params = st.multiselect(
    "Select parameters to display",
    available_params,
    default=available_params,
)

if not display_params:
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# Tab 1: Merged chart
# ══════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.subheader("\U0001f4c8 Weather Data Trends")

    # Build a single merged figure with all selected parameters
    n_params = len(display_params)
    fig = make_subplots(
        rows=n_params, cols=1,
        subplot_titles=[
            f"{PARAM_LABELS[p]} ({PARAM_UNITS[p]})" for p in display_params
        ],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for i, p in enumerate(display_params):
        df = station_info["data"][p]
        row = i + 1
        color = PARAM_COLORS[p]
        label = PARAM_LABELS[p]

        # Bar — daily values
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["value_avg"],
                name=f"{label} (bar)",
                marker_color=color,
                marker_opacity=0.55,
                showlegend=False,
            ),
            row=row, col=1,
        )

        # Line — smoothed trend overlay
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value_avg"],
                mode="lines+markers",
                name=f"{label} (trend)",
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                showlegend=False,
            ),
            row=row, col=1,
        )

        # Y-axis label per subplot
        fig.update_yaxes(
            title_text=PARAM_UNITS[p],
            row=row, col=1,
            gridcolor="rgba(128,128,128,0.15)",
        )

    fig.update_layout(
        title=f"Weather Data — {station_info['name']} [{selected_sid}]",
        height=280 * n_params,
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")

    # ── Summary statistics ──
    st.divider()
    st.subheader("\U0001f4ca Summary Statistics")

    stats_data = []
    for p in display_params:
        if p not in station_info["data"]:
            continue
        df = station_info["data"][p]
        stats_data.append({
            "Parameter":   PARAM_LABELS[p],
            "Unit":        PARAM_UNITS[p],
            "Mean":        f"{df['value_avg'].mean():.2f}",
            "Min":         f"{df['value_min'].min():.2f}",
            "Max":         f"{df['value_max'].max():.2f}",
            "Days of data": len(df),
        })

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# Tab 2: Data table
# ══════════════════════════════════════════════════════════════════════════
with tab_table:
    st.subheader("\U0001f4cb Raw Daily Data")

    table_param = st.selectbox(
        "Select parameter",
        display_params,
        format_func=lambda p: PARAM_LABELS[p],
        key="table_param",
    )

    if table_param in station_info["data"]:
        df = station_info["data"][table_param]
        st.dataframe(
            df[["date", "value_avg", "value_min", "value_max", "count"]],
            width="stretch",
            hide_index=True,
        )
        st.caption(f"{len(df)} days of data")


# ══════════════════════════════════════════════════════════════════════════
# Tab 3: Export
# ══════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("\U0001f4be Data Export")

    export_param = st.selectbox(
        "Export parameter",
        display_params,
        format_func=lambda p: PARAM_LABELS[p],
        key="export_param",
    )
    export_resolution = st.radio(
        "Resolution",
        ["Daily", "Hourly"],
        horizontal=True,
    )

    if st.button("\U0001f4e5 Export as CSV", type="primary"):
        try:
            export_start = datetime.combine(start_date, datetime.min.time())
            export_end   = datetime.combine(end_date,   datetime.min.time())

            out_dir = PROJECT_DIR / "exports"
            out_dir.mkdir(exist_ok=True)
            output_path = (
                out_dir
                / f"{selected_sid}_{export_param}_"
                  f"{export_start.date()}_{export_end.date()}.csv"
            )
            result_path = export_to_csv(
                station_id=selected_sid,
                parameter=export_param,
                start_date=export_start,
                end_date=export_end,
                output_path=output_path,
                resolution="daily" if export_resolution == "Daily" else "hourly",
            )
            st.success(f"\u2705 Exported: {result_path}")

            # Preview exported content
            df_preview = pd.read_csv(result_path)
            st.dataframe(df_preview.head(20), width="stretch", hide_index=True)
            st.caption(f"File contains {len(df_preview)} rows — showing first 20")
        except Exception as e:
            st.error(f"Export failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Data source: Deutscher Wetterdienst (DWD) Open Data  |  "
    f"Database: {DB_PATH}  |  "
    "Project: Wetterdatenbank und Visualisierung"
)
