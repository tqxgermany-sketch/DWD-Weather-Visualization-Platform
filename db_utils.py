"""
Database query and export utilities
====================================
Provides functions for reading, aggregating, and exporting weather data
from the local SQLite database. Used by the UI layer and analysis modules.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

from dwd_fetcher import DB_PATH, log


def get_stations(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return all weather stations stored in the database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM stations", conn)
    conn.close()
    return df


def get_available_parameters(station_id: str, db_path: Path = DB_PATH) -> list[str]:
    """Return the list of parameters that have data for a given station."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT parameter FROM observations WHERE station_id = ?",
        (station_id,)
    )
    params = [row[0] for row in cursor.fetchall()]
    conn.close()
    return params


def get_date_range(station_id: str, parameter: str, db_path: Path = DB_PATH) -> tuple:
    """Return the data time range (min, max) for a given station and parameter."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MIN(timestamp), MAX(timestamp)
        FROM observations
        WHERE station_id = ? AND parameter = ?
    """, (station_id, parameter))
    row = cursor.fetchone()
    conn.close()
    return (row[0], row[1]) if row[0] else (None, None)


def query_daily_avg(
    station_id: str,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Query daily averages (aggregates hourly data to daily).

    Returns: DataFrame with date / value_avg / value_min / value_max / count
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            SUBSTR(timestamp, 1, 10) AS date,
            AVG(value)               AS value_avg,
            MIN(value)               AS value_min,
            MAX(value)               AS value_max,
            COUNT(value)             AS count
        FROM observations
        WHERE station_id = ?
          AND parameter  = ?
          AND timestamp >= ?
          AND timestamp <= ?
        GROUP BY SUBSTR(timestamp, 1, 10)
        ORDER BY date
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            station_id,
            parameter,
            start_date.strftime("%Y-%m-%d %H:%M"),
            end_date.strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.close()
    return df


def export_to_csv(
    station_id: str,
    parameter: str,
    start_date: datetime,
    end_date: datetime,
    output_path: Path,
    resolution: str = "hourly",
    db_path: Path = DB_PATH,
) -> Path:
    """
    Export query results as a CSV file.

    Args:
        resolution: "hourly" or "daily"

    Returns:
        Output file path
    """
    from dwd_fetcher import query_data

    if resolution == "daily":
        df = query_daily_avg(station_id, parameter, start_date, end_date, db_path)
    else:
        df = query_data(station_id, parameter, start_date, end_date, db_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    log.info("Data exported: %s (%d rows)", output_path, len(df))
    return output_path


def get_monthly_stats(
    station_id: str,
    parameter: str,
    year: int,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return monthly statistics (avg/min/max) for a given station, parameter, and year."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            SUBSTR(timestamp, 1, 7)  AS month,
            AVG(value)               AS avg,
            MIN(value)               AS min,
            MAX(value)               AS max
        FROM observations
        WHERE station_id = ?
          AND parameter  = ?
          AND SUBSTR(timestamp, 1, 4) = ?
        GROUP BY SUBSTR(timestamp, 1, 7)
        ORDER BY month
    """

    df = pd.read_sql_query(query, conn, params=(station_id, parameter, str(year)))
    conn.close()
    return df
