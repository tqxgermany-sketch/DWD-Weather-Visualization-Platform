"""
DWD Climate Data Center — Data Fetch Module
============================================
Features:
  - Find nearest German weather station by lat/lon (auto-select stations with actual data)
  - Fetch historical weather data (temperature, precipitation, wind, sunshine)
  - Save data to local SQLite database
  - Supported period: 2015-01-01 ~ 2025-12-31

Dependencies:
  pip install wetterdienst geopy pandas

Note: This module uses an aiohttp SSL monkey-patch to work around
Windows certificate verification issues.
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

# ─── SSL fix (bypasses Windows certificate verification issues) ─────────────
def _patch_ssl():
    """Inject a non-SSL-verifying aiohttp client into wetterdienst HTTPFileSystem."""
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

# ─── Logging config ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress wetterdienst's own verbose logs
logging.getLogger("wetterdienst").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
DB_PATH      = PROJECT_DIR / "weather_data.db"
CACHE_DIR    = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DATE_MIN = datetime(2015, 1, 1)
DATE_MAX = datetime(2025, 12, 31, 23, 59)

# DWD parameter mapping: module key -> (resolution, dataset_api_name, parameter_name, url_path)
# dataset_api_name: name used in wetterdienst API
# url_path:         actual path fragment on DWD FTP
PARAMETER_MAP = {
    "temperature":   ("hourly", "temperature_air",  "temperature_air_mean_2m", "air_temperature"),
    "precipitation": ("hourly", "precipitation",    "precipitation_height",    "precipitation"),
    "wind_speed":    ("hourly", "wind",              "wind_speed",              "wind"),
    "sunshine":      ("hourly", "sun",               "sunshine_duration",       "sun"),
}

# ─── Global DWD Settings ────────────────────────────────────────────────────
def _make_settings() -> Settings:
    return Settings(
        ts_humanize=True,
        ts_shape="long",
        cache_dir=CACHE_DIR,
    )


# ─── Database initialization ────────────────────────────────────────────────
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create the database and tables (if they don't already exist)."""
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
    log.info("Database ready: %s", db_path)
    return conn


# ─── Station search (with data-availability filtering) ──────────────────────
def find_nearest_station(
    lat: float,
    lon: float,
    parameter: str = "temperature",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find the nearest German weather stations by lat/lon.

    Args:
        lat, lon: Target coordinates
        parameter: Weather parameter (from PARAMETER_MAP keys)
        top_n: Max number of candidate stations to return

    Returns:
        DataFrame with station_id / name / lat / lon / distance_km
    """
    if parameter not in PARAMETER_MAP:
        raise ValueError(
            f"Unsupported parameter: {parameter}, available: {list(PARAMETER_MAP.keys())}"
        )

    log.info("Searching nearest stations (%s), coords: (%.4f, %.4f)", parameter, lat, lon)

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
        raise ValueError(f"No weather station found within 200 km of ({lat}, {lon})")

    # Convert to pandas, compute distances, sort
    df = df_pl.to_pandas()
    df = df.drop_duplicates(subset=["station_id"])
    df["distance_km"] = df.apply(
        lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km,
        axis=1,
    )
    df = df.sort_values("distance_km").head(top_n)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    result = df[["station_id", "name", "lat", "lon", "distance_km"]].reset_index(drop=True)
    log.info(
        "Found %d candidate stations, nearest: %s (%.1f km)",
        len(result), result.iloc[0]["name"], result.iloc[0]["distance_km"]
    )
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
    Find the nearest station that actually has data for the requested time period.

    Returns:
        dict with station_id / name / lat / lon / distance_km
    """
    import re

    res, dataset, _, url_path = PARAMETER_MAP[parameter]
    settings = _make_settings()

    # Get candidate stations
    req = DwdObservationRequest(
        parameters=[(res, dataset)],
        start_date=DATE_MIN,
        end_date=DATE_MAX,
        periods=[Period.HISTORICAL, Period.RECENT],
        settings=settings,
    )

    stations_result = req.filter_by_distance(
        latlon=(lat, lon), distance=search_radius_km
    )
    df_pl = stations_result.df

    if df_pl.is_empty():
        raise ValueError(
            f"No weather station found within {search_radius_km} km of ({lat}, {lon})"
        )

    df = df_pl.to_pandas().drop_duplicates(subset=["station_id"])
    df["distance_km"] = df.apply(
        lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km,
        axis=1,
    )
    df = df.sort_values("distance_km").head(max_candidates)

    # Query file listings to find which stations have data covering the target period
    from wetterdienst.util.network import list_remote_files_fsspec
    from wetterdienst.metadata.cache import CacheExpiry

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
        recent_files = list_remote_files_fsspec(
            base_url_recent, settings=settings, cache_expiry=CacheExpiry.METAINDEX
        )
    except Exception:
        recent_files = []

    try:
        hist_files = list_remote_files_fsspec(
            base_url_hist, settings=settings, cache_expiry=CacheExpiry.METAINDEX
        )
    except Exception:
        hist_files = []

    all_files = recent_files + hist_files

    def station_has_data(station_id: str) -> bool:
        """Check whether this station has files covering the target period."""
        sid = station_id.zfill(5)
        for f in all_files:
            if sid not in f:
                continue
            fname = f.split("/")[-1]
            if fname.endswith("_akt.zip"):
                # Recent file: covers the last ~500 days, always overlaps 2015+
                return True
            # Historical file: extract date range
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
                "Selected station: [%s] %s (%.1f km)",
                sid, row["name"], row["distance_km"]
            )
            return {
                "station_id":  sid,
                "name":        row["name"],
                "lat":         float(row["latitude"]),
                "lon":         float(row["longitude"]),
                "distance_km": float(row["distance_km"]),
            }

    # Fallback: return the nearest station even if data availability is uncertain
    row = df.iloc[0]
    log.warning(
        "No station with confirmed data found; using nearest: [%s] %s",
        row["station_id"], row["name"]
    )
    return {
        "station_id":  str(row["station_id"]),
        "name":        row["name"],
        "lat":         float(row["latitude"]),
        "lon":         float(row["longitude"]),
        "distance_km": float(row["distance_km"]),
    }


# ─── Data fetch ─────────────────────────────────────────────────────────────
def fetch_weather_data(
    station_id: str,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Fetch weather data from DWD CDC for a given station, parameter, and time range.

    Returns:
        pandas DataFrame with timestamp (str) / value (float) columns.
        timestamp format: "YYYY-MM-DD HH:MM"
    """
    if parameter not in PARAMETER_MAP:
        raise ValueError(
            f"Unsupported parameter: {parameter}, available: {list(PARAMETER_MAP.keys())}"
        )

    start_date = max(start_date, DATE_MIN)
    end_date   = min(end_date,   DATE_MAX)

    log.info(
        "Fetching data | station: %s | parameter: %s | %s ~ %s",
        station_id, parameter,
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
    )

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
        log.warning("No data retrieved (station: %s, parameter: %s)", station_id, parameter)
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.concat(all_rows, ignore_index=True)

    # Filter to the target time range
    df["date"] = pd.to_datetime(df["date"], utc=True)
    mask = (df["date"] >= pd.Timestamp(start_date, tz="UTC")) & \
           (df["date"] <= pd.Timestamp(end_date,   tz="UTC"))
    df = df[mask].copy()

    # Standardize output
    df_clean = pd.DataFrame({
        "timestamp": df["date"].dt.strftime("%Y-%m-%d %H:%M"),
        "value":     df["value"].astype(float),
    })
    df_clean = df_clean.dropna(subset=["value"]).reset_index(drop=True)

    log.info("Fetch complete, %d valid records", len(df_clean))
    return df_clean


# ─── Save to database ──────────────────────────────────────────────────────
def save_to_db(
    conn: sqlite3.Connection,
    station_id: str,
    station_name: str,
    lat: float,
    lon: float,
    parameter: str,
    df: pd.DataFrame,
) -> int:
    """Write data to SQLite; duplicates are silently ignored."""
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
    log.info("Written to DB: %d new records (parameter: %s)", inserted, parameter)
    return inserted


# ─── High-level wrapper: one-click fetch and store ──────────────────────────
def fetch_and_store(
    lat: float,
    lon: float,
    parameters: list[str],
    start_date: datetime,
    end_date: datetime,
    db_path: Path = DB_PATH,
) -> dict:
    """
    Main entry point: auto-find nearest station by coordinates,
    fetch all specified parameters, and store in the local database.

    Args:
        lat, lon:    Target coordinates (within Germany)
        parameters:  List of parameter keys, e.g. ["temperature", "precipitation"]
        start_date:  Start of time range
        end_date:    End of time range
        db_path:     Path to SQLite database

    Returns:
        dict with station_id / station_name / distance_km / records_inserted
    """
    conn = init_db(db_path)

    # 1. Find nearest station with confirmed data
    nearest      = find_nearest_station_with_data(
        lat, lon, parameters[0], start_date, end_date
    )
    station_id   = nearest["station_id"]
    station_name = nearest["name"]
    s_lat        = nearest["lat"]
    s_lon        = nearest["lon"]
    distance_km  = nearest["distance_km"]

    log.info(
        "Using station: [%s] %s (distance %.1f km)",
        station_id, station_name, distance_km
    )

    # 2. Fetch and store each parameter
    total_inserted = 0
    for param in parameters:
        try:
            df = fetch_weather_data(station_id, param, start_date, end_date)
            if not df.empty:
                n = save_to_db(
                    conn, station_id, station_name, s_lat, s_lon, param, df
                )
                total_inserted += n
        except Exception as e:
            log.error("Parameter %s fetch failed: %s", param, e)

    conn.close()

    return {
        "station_id":       station_id,
        "station_name":     station_name,
        "distance_km":      distance_km,
        "records_inserted": total_inserted,
    }


# ─── Database query interface ───────────────────────────────────────────────
def query_data(
    station_id: str,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Query stored weather data from the local database."""
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

    log.info(
        "Query result: %d records (station: %s, parameter: %s)",
        len(df), station_id, parameter
    )
    return df


# ─── CLI quick test entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DWD Data Fetch Module — Quick Test")
    print("=" * 60)

    # Test coordinates: Munich city center
    TEST_LAT   = 48.137
    TEST_LON   = 11.576
    TEST_START = datetime(2024, 1, 1)
    TEST_END   = datetime(2024, 1, 7)

    print(f"\nTest location: Munich ({TEST_LAT}, {TEST_LON})")
    print(f"Test period:   {TEST_START.date()} ~ {TEST_END.date()}")
    print(f"Test param:    temperature\n")

    result = fetch_and_store(
        lat        = TEST_LAT,
        lon        = TEST_LON,
        parameters = ["temperature"],
        start_date = TEST_START,
        end_date   = TEST_END,
    )

    print("\n─── Result ───────────────────────────")
    print(f"  Station:  [{result['station_id']}] {result['station_name']}")
    print(f"  Distance: {result['distance_km']:.1f} km")
    print(f"  Records:  {result['records_inserted']}")

    # Verify: read back from database
    df_check = query_data(
        station_id = result["station_id"],
        parameter  = "temperature",
        start_date = TEST_START,
        end_date   = TEST_END,
    )

    if not df_check.empty:
        print(f"\n─── Data preview (first 8 rows) ───────")
        print(df_check.head(8).to_string(index=False))
    else:
        print("\nDatabase is empty — check network or DWD API status")

    print(f"\nDatabase file: {DB_PATH}")
