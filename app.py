"""
DWD 气象数据可视化平台 — Streamlit UI
=====================================
功能:
  - 坐标/城市定位 → 自动匹配最近气象站
  - 时间范围 + 参数选择 → 一键拉取
  - Plotly 交互图表（气温/降水/风力/日照）
  - CSV 导出
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 确保当前目录在 sys.path 中，以便导入项目模块
PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dwd_fetcher import (
    fetch_weather_data,
    fetch_and_store,
    find_nearest_station_with_data,
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
    get_monthly_stats,
)

# ─── 页面配置 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DWD 气象数据平台",
    page_icon="🌤️",
    layout="wide",
)

# ─── 中文字段标签 ─────────────────────────────────────────────────────────
PARAM_LABELS = {
    "temperature":   "气温",
    "precipitation": "降水",
    "wind_speed":    "风速",
    "sunshine":      "日照",
}

PARAM_UNITS = {
    "temperature":   "°C",
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

# ─── 标题 ─────────────────────────────────────────────────────────────────
st.title("🌤️ DWD 德国气象数据可视化平台")
st.caption(f"数据来源: Deutscher Wetterdienst (DWD) Climate Data Center  |  时间范围: {DATE_MIN.date()} ~ {DATE_MAX.date()}")

# ══════════════════════════════════════════════════════════════════════════
# 侧边栏 — 配置面板
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 查询配置")

    # ── 定位方式 ──
    st.subheader("📍 位置")
    location_mode = st.radio(
        "输入方式",
        ["坐标 (纬度/经度)", "德国城市名"],
        horizontal=True,
    )

    if location_mode == "坐标 (纬度/经度)":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("纬度 (Lat)", min_value=47.0, max_value=55.5,
                                  value=48.137, step=0.01, format="%.4f",
                                  help="德国纬度范围: 47° ~ 55.5°")
        with col2:
            lon = st.number_input("经度 (Lon)", min_value=5.5, max_value=15.5,
                                  value=11.576, step=0.01, format="%.4f",
                                  help="德国经度范围: 5.5° ~ 15.5°")
    else:
        # 预置德国主要城市坐标
        GERMAN_CITIES = {
            "柏林 (Berlin)":           (52.520, 13.405),
            "汉堡 (Hamburg)":          (53.551, 9.994),
            "慕尼黑 (München)":        (48.137, 11.576),
            "科隆 (Köln)":             (50.938, 6.960),
            "法兰克福 (Frankfurt)":    (50.111, 8.682),
            "斯图加特 (Stuttgart)":    (48.776, 9.178),
            "杜塞尔多夫 (Düsseldorf)": (51.228, 6.773),
            "莱比锡 (Leipzig)":        (51.340, 12.375),
            "多特蒙德 (Dortmund)":     (51.514, 7.465),
            "不莱梅 (Bremen)":         (53.079, 8.801),
            "德累斯顿 (Dresden)":      (51.051, 13.738),
            "汉诺威 (Hannover)":       (52.376, 9.741),
            "纽伦堡 (Nürnberg)":       (49.452, 11.077),
        }
        city = st.selectbox("选择城市", list(GERMAN_CITIES.keys()))
        lat, lon = GERMAN_CITIES[city]
        st.caption(f"坐标: ({lat:.4f}, {lon:.4f})")

    st.divider()

    # ── 时间范围 ──
    st.subheader("📅 时间范围")
    start_date = st.date_input(
        "开始日期",
        value=datetime(2024, 1, 1),
        min_value=DATE_MIN.date(),
        max_value=DATE_MAX.date(),
    )
    end_date = st.date_input(
        "结束日期",
        value=datetime(2024, 1, 31),
        min_value=DATE_MIN.date(),
        max_value=DATE_MAX.date(),
    )
    if start_date > end_date:
        st.error("开始日期不能晚于结束日期")

    st.divider()

    # ── 参数选择 ──
    st.subheader("📊 气象参数")
    params_selected = []
    for p, label in PARAM_LABELS.items():
        if st.checkbox(f"{label} ({p})", value=(p == "temperature")):
            params_selected.append(p)

    st.divider()

    # ── 搜索半径 ──
    search_radius = st.slider(
        "气象站搜索半径 (km)",
        min_value=10, max_value=300, value=100, step=10,
        help="在目标坐标指定半径内查找最近气象站",
    )

    st.divider()

    # ── 执行按钮 ──
    fetch_btn = st.button(
        "🚀 拉取数据",
        type="primary",
        width="stretch",
        disabled=(len(params_selected) == 0 or start_date > end_date),
    )

    # ── 已有数据提示 ──
    existing_params = []
    try:
        stations_df = get_stations()
        if not stations_df.empty:
            # 检查每个站点是否有选中参数的数据
            for _, station_row in stations_df.iterrows():
                sid = station_row["station_id"]
                avail = get_available_parameters(sid)
                for p in params_selected:
                    date_from, date_to = get_date_range(sid, p)
                    if date_from:
                        existing_params.append((sid, station_row["name"], p, date_from, date_to))
    except Exception:
        pass

    if existing_params:
        st.divider()
        st.subheader("💾 本地已有数据")
        seen = set()
        for sid, sname, p, dfrom, dto in existing_params:
            key = (sid, p)
            if key not in seen:
                seen.add(key)
                st.caption(f"• {sname} [{sid}] — {PARAM_LABELS[p]}: {dfrom} ~ {dto}")


# ══════════════════════════════════════════════════════════════════════════
# 主区域 — 数据展示
# ══════════════════════════════════════════════════════════════════════════

# ── Tab 结构 ──
tab_chart, tab_table, tab_export = st.tabs(["📈 可视化", "📋 数据表", "💾 导出"])

# ── 数据拉取逻辑 ──
if fetch_btn:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.min.time())

    with st.spinner("正在查找最近气象站并拉取数据..."):
        try:
            result = fetch_and_store(
                lat=lat, lon=lon,
                parameters=params_selected,
                start_date=start_dt,
                end_date=end_dt,
            )
            st.session_state["last_result"] = result
            st.session_state["last_params"] = params_selected
            st.session_state["last_start"]  = start_dt
            st.session_state["last_end"]    = end_dt
            st.session_state["data_loaded"] = True

            st.success(
                f"✅ 数据拉取完成！\n\n"
                f"气象站: **{result['station_name']}** [{result['station_id']}]\n\n"
                f"距离: {result['distance_km']:.1f} km  |  写入: {result['records_inserted']} 条记录"
            )
        except Exception as e:
            st.error(f"❌ 拉取失败: {e}")
            st.session_state["data_loaded"] = False


# ── 检查是否有可展示的数据 ──
def _load_data():
    """从数据库加载当前选中参数的数据"""
    all_data = {}
    stations_df = get_stations()
    if stations_df.empty:
        return all_data

    for _, station_row in stations_df.iterrows():
        sid = station_row["station_id"]
        for p in params_selected:
            date_from, date_to = get_date_range(sid, p)
            if date_from is None:
                continue
            df = query_daily_avg(
                sid, p,
                datetime.strptime(date_from[:10], "%Y-%m-%d"),
                datetime.strptime(date_to[:10], "%Y-%m-%d"),
            )
            if not df.empty:
                sid = str(sid)  # ensure string
                if sid not in all_data:
                    all_data[sid] = {
                        "name":  station_row["name"],
                        "lat":   station_row["lat"],
                        "lon":   station_row["lon"],
                        "data":  {},
                    }
                all_data[sid]["data"][p] = df
    return all_data


all_data = _load_data()

if not all_data:
    st.info("👈 请先在左侧面板设置参数并点击「拉取数据」，或调整时间范围以匹配已有数据。")
    st.stop()

# ── 选择展示的站点 ──
station_ids   = list(all_data.keys())
station_names = [f"[{sid}] {all_data[sid]['name']}" for sid in station_ids]

selected_station_label = st.selectbox("选择气象站", station_names)
selected_sid = station_ids[station_names.index(selected_station_label)]
station_info = all_data[selected_sid]

# 站点信息卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("站点名称", station_info["name"])
with col2:
    st.metric("站点 ID", selected_sid)
with col3:
    st.metric("纬度", f"{station_info['lat']:.4f}°N")
with col4:
    st.metric("经度", f"{station_info['lon']:.4f}°E")

# ── 参数选择（展示部分） ──
available_params = [p for p in params_selected if p in station_info["data"]]
if not available_params:
    st.warning("当前选中参数在数据库中无数据，请先拉取。")
    st.stop()

display_params = st.multiselect(
    "选择要显示的气象参数",
    available_params,
    default=available_params,
)

if not display_params:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# Tab 1: 可视化图表
# ══════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.subheader("📈 气象数据趋势图")

    controls_col1, controls_col2 = st.columns([3, 1])
    with controls_col1:
        chart_mode = st.radio(
            "图表模式",
            ["合并视图", "分列视图"],
            horizontal=True,
        )
    with controls_col2:
        show_daily = st.checkbox("显示日均值", value=True)

    if chart_mode == "合并视图":
        # ── 气温图（单独，因为范围不同） ──
        temp_plotted = False
        for p in display_params:
            if p == "temperature" and p in station_info["data"]:
                temp_plotted = True
                df = station_info["data"][p]
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df["date"],
                    y=df["value_avg"],
                    mode="lines",
                    name="气温 (°C)",
                    line=dict(color=PARAM_COLORS[p], width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"],
                    y=df["value_min"],
                    mode="lines",
                    name="最低温",
                    line=dict(color=PARAM_COLORS[p], width=0.5, dash="dot"),
                    showlegend=True,
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"],
                    y=df["value_max"],
                    mode="lines",
                    name="最高温",
                    line=dict(color=PARAM_COLORS[p], width=0.5, dash="dot"),
                    showlegend=True,
                ))
                # 填充最低-最高区间
                fig.add_trace(go.Scatter(
                    x=list(df["date"]) + list(df["date"])[::-1],
                    y=list(df["value_min"]) + list(df["value_max"])[::-1],
                    fill="toself",
                    fillcolor=f"rgba(231, 76, 60, 0.1)",
                    line=dict(width=0),
                    name="波动范围",
                    showlegend=False,
                ))

                fig.update_layout(
                    title=f"气温变化趋势 — {station_info['name']} [{selected_sid}]",
                    xaxis_title="日期",
                    yaxis_title="温度 (°C)",
                    hovermode="x unified",
                    height=400,
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(fig, width="stretch")

        # ── 其他参数子图 ──
        other_params = [p for p in display_params if p != "temperature" and p in station_info["data"]]
        if other_params:
            n = len(other_params)
            fig2 = make_subplots(
                rows=n, cols=1,
                subplot_titles=[f"{PARAM_LABELS[p]} ({PARAM_UNITS[p]})" for p in other_params],
                shared_xaxes=True,
                vertical_spacing=0.08,
            )

            for i, p in enumerate(other_params):
                df = station_info["data"][p]
                fig2.add_trace(
                    go.Bar(
                        x=df["date"],
                        y=df["value_avg"] if show_daily else df["value_avg"],
                        name=PARAM_LABELS[p],
                        marker_color=PARAM_COLORS[p],
                        showlegend=False,
                    ),
                    row=i + 1, col=1,
                )

            fig2.update_layout(
                height=250 * n,
                hovermode="x unified",
                margin=dict(l=40, r=40, t=40, b=40),
            )
            # 单张图时调整
            if n == 1:
                fig2.update_layout(height=350)

            if not temp_plotted:
                fig2.update_layout(
                    title=f"气象数据 — {station_info['name']} [{selected_sid}]",
                )

            st.plotly_chart(fig2, width="stretch")

    else:
        # ── 分列视图：每个参数独立图表 ──
        for p in display_params:
            if p not in station_info["data"]:
                continue
            df = station_info["data"][p]

            fig = go.Figure()

            if p == "temperature":
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["value_avg"],
                    mode="lines+markers",
                    name="日均温",
                    line=dict(color=PARAM_COLORS[p], width=2),
                    marker=dict(size=3),
                ))
                fig.add_trace(go.Scatter(
                    x=list(df["date"]) + list(df["date"])[::-1],
                    y=list(df["value_min"]) + list(df["value_max"])[::-1],
                    fill="toself",
                    fillcolor=f"rgba(231, 76, 60, 0.1)",
                    line=dict(width=0),
                    name="最低-最高区间",
                ))
            elif p == "precipitation":
                fig.add_trace(go.Bar(
                    x=df["date"], y=df["value_avg"],
                    name="降水量",
                    marker_color=PARAM_COLORS[p],
                ))
            elif p == "sunshine":
                fig.add_trace(go.Bar(
                    x=df["date"], y=df["value_avg"],
                    name="日照时长",
                    marker_color=PARAM_COLORS[p],
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["value_avg"],
                    mode="lines",
                    name=PARAM_LABELS[p],
                    line=dict(color=PARAM_COLORS[p], width=2),
                ))

            fig.update_layout(
                title=f"{PARAM_LABELS[p]} — {station_info['name']}",
                xaxis_title="日期",
                yaxis_title=f"{PARAM_LABELS[p]} ({PARAM_UNITS[p]})",
                hovermode="x unified",
                height=350,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, width="stretch")

    # ── 统计摘要 ──
    st.divider()
    st.subheader("📊 统计摘要")

    stats_data = []
    for p in display_params:
        if p not in station_info["data"]:
            continue
        df = station_info["data"][p]
        stats_data.append({
            "参数":     PARAM_LABELS[p],
            "单位":     PARAM_UNITS[p],
            "日均值":   f"{df['value_avg'].mean():.2f}",
            "最小值":   f"{df['value_min'].min():.2f}",
            "最大值":   f"{df['value_max'].max():.2f}",
            "数据天数": len(df),
        })

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# Tab 2: 数据表
# ══════════════════════════════════════════════════════════════════════════
with tab_table:
    st.subheader("📋 原始日数据")

    table_param = st.selectbox(
        "选择参数查看",
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
        st.caption(f"共 {len(df)} 天数据")


# ══════════════════════════════════════════════════════════════════════════
# Tab 3: 导出
# ══════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("💾 数据导出")

    export_param = st.selectbox(
        "导出参数",
        display_params,
        format_func=lambda p: PARAM_LABELS[p],
        key="export_param",
    )
    export_resolution = st.radio("数据精度", ["日级", "小时级"], horizontal=True)

    if st.button("📥 导出为 CSV", type="primary"):
        try:
            export_start = datetime.combine(start_date, datetime.min.time())
            export_end   = datetime.combine(end_date,   datetime.min.time())

            output_path = PROJECT_DIR / "exports" / f"{selected_sid}_{export_param}_{export_start.date()}_{export_end.date()}.csv"
            result_path = export_to_csv(
                station_id=selected_sid,
                parameter=export_param,
                start_date=export_start,
                end_date=export_end,
                output_path=output_path,
                resolution="daily" if export_resolution == "日级" else "hourly",
            )
            st.success(f"✅ 已导出: {result_path}")

            # 预览导出内容
            df_preview = pd.read_csv(result_path)
            st.dataframe(df_preview.head(20), width="stretch", hide_index=True)
            st.caption(f"文件共 {len(df_preview)} 行，此处显示前 20 行")
        except Exception as e:
            st.error(f"导出失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 底部
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "数据来源: Deutscher Wetterdienst (DWD) Open Data  |  "
    f"数据库: {DB_PATH}  |  "
    "项目: Wetterdatenbank und Visualisierung"
)
