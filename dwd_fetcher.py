"""
DWD Climate Data Center - 数据拉取模块
========================================
功能:
  - 根据经纬度坐标查找最近德国气象站（自动匹配有数据的站点）
  - 拉取历史气象数据 (气温、降水、风力、日照)
  - 将数据保存到本地 SQLite 数据库
  - 支持时间范围: 2015-01-01 ~ 2025-12-31

依赖:
  pip install wetterdienst geopy pandas

注意: 本模块使用 aiohttp SSL monkey-patch 解决 Windows 证书验证问题
"""

from __future__ import annotations

import ssl
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import aiohttp
import pandas as pd
from geopy.distance import geodesic
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst.metadata.period import Period
from wetterdienst.util import network as _net_module

# ─── SSL 修复（解决 Windows 证书验证失败问题）──────────────────────────────────
def _patch_ssl():
    """向 wetterdienst HTTPFileSystem 注入无 SSL 验证的 aiohttp client"""
    async def _no_ssl_get_client(**kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs.pop("ssl", None)
        connector = aiohttp.TCPConnector(ssl=ctx)
        return aiohttp.ClientSession(connector=connector, **kwargs)

    _orig_init = _net_module.HTTPFileSystem.__init__

    def _patched_init(
        self, /, *, use_listings_cache, listings_expiry_time,
        listings_cache_location=None, use_certifi=False, **kwargs
    ):
        kwargs["get_client"] = _no_ssl_get_client
        _orig_init(
            self,
            use_listings_cache=use_listings_cache,
            listings_expiry_time=listings_expiry_time,
            listings_cache_location=listings_cache_location,
            use_certifi=use_certifi,
            **kwargs,
        )

    _net_module.HTTPFileSystem.__init__ = _patched_init


_patch_ssl()

# ─── 日志配置 ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# 屏蔽 wetterdienst 自身的冗余日志
logging.getLogger("wetterdienst").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ─── 常量 ───────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
DB_PATH      = PROJECT_DIR / "weather_data.db"
CACHE_DIR    = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DATE_MIN = datetime(2015, 1, 1)
DATE_MAX = datetime(2025, 12, 31, 23, 59)

# DWD 参数映射: 本模块名称 -> (resolution, dataset_api_name, parameter_name, url_path)
# dataset_api_name: wetterdienst API 中使用的名称
# url_path:         DWD FTP 目录中的实际路径片段
PARAMETER_MAP = {
    "temperature":   ("hourly", "temperature_air",  "temperature_air_mean_2m", "air_temperature"),
    "precipitation": ("hourly", "precipitation",    "precipitation_height",    "precipitation"),
    "wind_speed":    ("hourly", "wind",              "wind_speed",              "wind"),
    "sunshine":      ("hourly", "sun",               "sunshine_duration",       "sun"),
}

# ─── 全局 DWD Settings ───────────────────────────────────────────────────────
def _make_settings() -> Settings:
    return Settings(
        ts_humanize=True,
        ts_shape="long",
        cache_dir=CACHE_DIR,
    )


# ─── 数据库初始化 ────────────────────────────────────────────────────────────
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """创建数据库及数据表（如果不存在则新建）"""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            station_id  TEXT PRIMARY KEY,
            name        TEXT,
            lat         REAL,
            lon         REAL,
            altitude    REAL,
            state       TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id  TEXT,
            timestamp   TEXT,
            parameter   TEXT,
            value       REAL,
            UNIQUE(station_id, timestamp, parameter)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_obs_station_time
        ON observations (station_id, timestamp)
    """)

    conn.commit()
    log.info("数据库就绪: %s", db_path)
    return conn


# ─── 站点查询（含数据可用性过滤） ──────────────────────────────────────────────
def find_nearest_station(
    lat: float,
    lon: float,
    parameter: str = "temperature",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    根据经纬度找最近且有数据的德国气象站

    参数:
        lat, lon: 目标坐标
        parameter: 气象参数（从 PARAMETER_MAP 键中选）
        top_n: 最多返回候选站数量

    返回:
        DataFrame，含 station_id / name / lat / lon / distance_km
    """
    if parameter not in PARAMETER_MAP:
        raise ValueError(f"不支持的参数: {parameter}，可选: {list(PARAMETER_MAP.keys())}")

    log.info("查询最近气象站 (%s)，坐标: (%.4f, %.4f)", parameter, lat, lon)

    res, dataset, _ , _ = PARAMETER_MAP[parameter]
    settings = _make_settings()

    req = DwdObservationRequest(
        parameters=[(res, dataset)],
        start_date=DATE_MIN,
        end_date=DATE_MAX,
        periods=[Period.HISTORICAL, Period.RECENT],
        settings=settings,
    )

    stations_result = req.filter_by_distance(latlon=(lat, lon), distance=200)
    df_pl = stations_result.df

    if df_pl.is_empty():
        raise ValueError(f"在坐标 ({lat}, {lon}) 200km 范围内未找到气象站")

    # 转为 pandas，计算距离，排序
    df = df_pl.to_pandas()
    df = df.drop_duplicates(subset=["station_id"])
    df["distance_km"] = df.apply(
        lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km,
        axis=1,
    )
    df = df.sort_values("distance_km").head(top_n)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    result = df[["station_id", "name", "lat", "lon", "distance_km"]].reset_index(drop=True)
    log.info("找到 %d 个候选站点，最近: %s (%.1f km)",
             len(result), result.iloc[0]["name"], result.iloc[0]["distance_km"])
    return result


def find_nearest_station_with_data(
    lat: float,
    lon: float,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
    search_radius_km: int = 200,
    max_candidates: int = 20,
) -> dict:
    """
    找最近且在指定时段内有实际数据的气象站

    返回:
        dict，含 station_id / name / lat / lon / distance_km
    """
    from wetterdienst.metadata.cache import CacheExpiry
    import re

    res, dataset, _, url_path = PARAMETER_MAP[parameter]
    settings = _make_settings()

    # 获取候选站点
    req = DwdObservationRequest(
        parameters=[(res, dataset)],
        start_date=DATE_MIN,
        end_date=DATE_MAX,
        periods=[Period.HISTORICAL, Period.RECENT],
        settings=settings,
    )

    stations_result = req.filter_by_distance(latlon=(lat, lon), distance=search_radius_km)
    df_pl = stations_result.df

    if df_pl.is_empty():
        raise ValueError(f"在坐标 ({lat}, {lon}) {search_radius_km}km 范围内未找到气象站")

    df = df_pl.to_pandas().drop_duplicates(subset=["station_id"])
    df["distance_km"] = df.apply(
        lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km,
        axis=1,
    )
    df = df.sort_values("distance_km").head(max_candidates)

    # 查询各站点文件，找出覆盖目标时段的站点
    from wetterdienst.util.network import list_remote_files_fsspec
    from wetterdienst.metadata.cache import CacheExpiry

    # 构造 DWD 路径片段
    res, dataset, _, url_path = PARAMETER_MAP[parameter]
    base_url_recent = (
        f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"
        f"{res}/{url_path}/recent/"
    )
    base_url_hist = (
        f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"
        f"{res}/{url_path}/historical/"
    )

    try:
        recent_files = list_remote_files_fsspec(base_url_recent, settings=settings, cache_expiry=CacheExpiry.METAINDEX)
    except Exception:
        recent_files = []

    try:
        hist_files = list_remote_files_fsspec(base_url_hist, settings=settings, cache_expiry=CacheExpiry.METAINDEX)
    except Exception:
        hist_files = []

    all_files = recent_files + hist_files

    def station_has_data(station_id: str) -> bool:
        """检查该站点是否有覆盖目标时段的文件"""
        sid = station_id.zfill(5)
        for f in all_files:
            if sid not in f:
                continue
            fname = f.split("/")[-1]
            if fname.endswith("_akt.zip"):
                # recent 文件：最近 500 天，一定覆盖 2015+ 的大部分时段
                return True
            # 历史文件：提取日期范围
            m = re.search(r"_(\d{8})_(\d{8})_", fname)
            if m:
                file_start = datetime.strptime(m.group(1), "%Y%m%d")
                file_end   = datetime.strptime(m.group(2), "%Y%m%d")
                if file_start <= end_date and file_end >= start_date:
                    return True
        return False

    for _, row in df.iterrows():
        sid = str(row["station_id"])
        if station_has_data(sid):
            log.info(
                "选定气象站: [%s] %s (%.1f km)",
                sid, row["name"], row["distance_km"]
            )
            return {
                "station_id":  sid,
                "name":        row["name"],
                "lat":         float(row["latitude"]),
                "lon":         float(row["longitude"]),
                "distance_km": float(row["distance_km"]),
            }

    # 回退：返回最近站点
    row = df.iloc[0]
    log.warning("未找到确认有数据的站点，使用最近站点: [%s] %s", row["station_id"], row["name"])
    return {
        "station_id":  str(row["station_id"]),
        "name":        row["name"],
        "lat":         float(row["latitude"]),
        "lon":         float(row["longitude"]),
        "distance_km": float(row["distance_km"]),
    }


# ─── 数据拉取 ────────────────────────────────────────────────────────────────
def fetch_weather_data(
    station_id: str,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    从 DWD CDC 拉取指定站点、参数、时段的气象数据

    返回:
        pandas DataFrame，含 timestamp (str) / value (float) 列
        timestamp 格式: "YYYY-MM-DD HH:MM"
    """
    if parameter not in PARAMETER_MAP:
        raise ValueError(f"不支持的参数: {parameter}，可选: {list(PARAMETER_MAP.keys())}")

    start_date = max(start_date, DATE_MIN)
    end_date   = min(end_date,   DATE_MAX)

    log.info("拉取数据 | 站点: %s | 参数: %s | %s ~ %s",
             station_id, parameter,
             start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    res, dataset, param_name, _ = PARAMETER_MAP[parameter]
    settings = _make_settings()

    req = DwdObservationRequest(
        parameters=[(res, dataset, param_name)],
        start_date=start_date,
        end_date=end_date,
        periods=[Period.HISTORICAL, Period.RECENT],
        settings=settings,
    )

    all_rows = []
    for vr in req.filter_by_station_id(station_id=[station_id]).values.query():
        df_part = vr.df.to_pandas()
        if not df_part.empty:
            all_rows.append(df_part)

    if not all_rows:
        log.warning("未获取到数据 (站点: %s, 参数: %s)", station_id, parameter)
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.concat(all_rows, ignore_index=True)

    # 过滤到目标时间范围内
    df["date"] = pd.to_datetime(df["date"], utc=True)
    mask = (df["date"] >= pd.Timestamp(start_date, tz="UTC")) & \
           (df["date"] <= pd.Timestamp(end_date,   tz="UTC"))
    df = df[mask].copy()

    # 标准化输出
    df_clean = pd.DataFrame({
        "timestamp": df["date"].dt.strftime("%Y-%m-%d %H:%M"),
        "value":     df["value"].astype(float),
    })
    df_clean = df_clean.dropna(subset=["value"]).reset_index(drop=True)

    log.info("拉取完成，共 %d 条有效记录", len(df_clean))
    return df_clean


# ─── 存入数据库 ──────────────────────────────────────────────────────────────
def save_to_db(
    conn: sqlite3.Connection,
    station_id: str,
    station_name: str,
    lat: float,
    lon: float,
    parameter: str,
    df: pd.DataFrame,
) -> int:
    """将数据写入 SQLite，重复数据自动忽略"""
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO stations (station_id, name, lat, lon)
        VALUES (?, ?, ?, ?)
    """, (station_id, station_name, lat, lon))

    records = [
        (station_id, row["timestamp"], parameter, row["value"])
        for _, row in df.iterrows()
    ]

    cur.executemany("""
        INSERT OR IGNORE INTO observations (station_id, timestamp, parameter, value)
        VALUES (?, ?, ?, ?)
    """, records)

    inserted = cur.rowcount
    conn.commit()
    log.info("写入数据库: %d 条新记录 (参数: %s)", inserted, parameter)
    return inserted


# ─── 高层封装：一键拉取并存储 ────────────────────────────────────────────────
def fetch_and_store(
    lat: float,
    lon: float,
    parameters: list[str],
    start_date: datetime,
    end_date: datetime,
    db_path: Path = DB_PATH,
) -> dict:
    """
    主入口：根据坐标自动找最近站点，拉取所有指定参数，存入本地数据库

    参数:
        lat, lon:    目标坐标 (德国境内)
        parameters:  要拉取的参数列表，如 ["temperature", "precipitation"]
        start_date:  开始时间
        end_date:    结束时间
        db_path:     SQLite 数据库路径

    返回:
        dict，含 station_id / station_name / distance_km / records_inserted
    """
    conn = init_db(db_path)

    # 1. 找最近且有数据的站点
    nearest      = find_nearest_station_with_data(lat, lon, parameters[0], start_date, end_date)
    station_id   = nearest["station_id"]
    station_name = nearest["name"]
    s_lat        = nearest["lat"]
    s_lon        = nearest["lon"]
    distance_km  = nearest["distance_km"]

    log.info("使用气象站: [%s] %s (距离 %.1f km)", station_id, station_name, distance_km)

    # 2. 逐参数拉取并存储
    total_inserted = 0
    for param in parameters:
        try:
            df = fetch_weather_data(station_id, param, start_date, end_date)
            if not df.empty:
                n = save_to_db(conn, station_id, station_name, s_lat, s_lon, param, df)
                total_inserted += n
        except Exception as e:
            log.error("参数 %s 拉取失败: %s", param, e)

    conn.close()

    return {
        "station_id":       station_id,
        "station_name":     station_name,
        "distance_km":      distance_km,
        "records_inserted": total_inserted,
    }


# ─── 数据库查询接口 ──────────────────────────────────────────────────────────
def query_data(
    station_id: str,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """从本地数据库查询已存储的气象数据"""
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT timestamp, value
        FROM observations
        WHERE station_id = ?
          AND parameter  = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
        """,
        conn,
        params=(
            station_id,
            parameter,
            start_date.strftime("%Y-%m-%d %H:%M"),
            end_date.strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.close()

    log.info("查询结果: %d 条记录 (站点: %s, 参数: %s)", len(df), station_id, parameter)
    return df


# ─── 命令行快速测试入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DWD 数据拉取模块 - 快速测试")
    print("=" * 60)

    # 测试坐标：慕尼黑市中心
    TEST_LAT   = 48.137
    TEST_LON   = 11.576
    TEST_START = datetime(2024, 1, 1)
    TEST_END   = datetime(2024, 1, 7)

    print(f"\n测试位置: 慕尼黑 ({TEST_LAT}, {TEST_LON})")
    print(f"测试时段: {TEST_START.date()} ~ {TEST_END.date()}")
    print(f"测试参数: temperature\n")

    result = fetch_and_store(
        lat        = TEST_LAT,
        lon        = TEST_LON,
        parameters = ["temperature"],
        start_date = TEST_START,
        end_date   = TEST_END,
    )

    print("\n─── 结果 ───────────────────────────")
    print(f"  气象站:   [{result['station_id']}] {result['station_name']}")
    print(f"  距离:     {result['distance_km']:.1f} km")
    print(f"  写入记录: {result['records_inserted']} 条")

    # 验证：从数据库读回
    df_check = query_data(
        station_id = result["station_id"],
        parameter  = "temperature",
        start_date = TEST_START,
        end_date   = TEST_END,
    )

    if not df_check.empty:
        print(f"\n─── 数据预览（前 8 行）───────────────")
        print(df_check.head(8).to_string(index=False))
    else:
        print("\n数据库为空，请检查网络或 DWD 接口状态")

    print(f"\n数据库文件: {DB_PATH}")
