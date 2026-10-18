"""
Database module for AgroPrice AI using SQLite.
Stores and manages user prediction history, input parameters, and model versions.
"""

import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT NOT NULL,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            market TEXT NOT NULL,
            variety TEXT,
            month INTEGER NOT NULL,
            year INTEGER NOT NULL,
            season TEXT,
            predicted_price REAL NOT NULL,
            price_range_low REAL,
            price_range_high REAL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
    """)

    conn.commit()
    conn.close()


def save_prediction(
    crop: str,
    state: str,
    district: str,
    market: str,
    month: int,
    year: int,
    predicted_price: float,
    variety: Optional[str] = "Standard",
    season: Optional[str] = None,
    price_range_low: Optional[float] = None,
    price_range_high: Optional[float] = None,
    model_version: Optional[str] = "v1.0.0"
) -> int:
    """Saves a prediction record to the local SQLite database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            crop, state, district, market, variety,
            month, year, season, predicted_price,
            price_range_low, price_range_high, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        crop, state, district, market, variety,
        month, year, season, round(predicted_price, 2),
        round(price_range_low, 2) if price_range_low is not None else None,
        round(price_range_high, 2) if price_range_high is not None else None,
        model_version
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_predictions(
    search_query: Optional[str] = None,
    crop_filter: Optional[str] = None,
    limit: int = 200
) -> pd.DataFrame:
    """Retrieves saved predictions with optional search query and crop filtering."""
    init_db()
    conn = get_db_connection()

    query = "SELECT * FROM predictions WHERE 1=1"
    params: List[Any] = []

    if crop_filter and crop_filter != "All":
        query += " AND LOWER(crop) = LOWER(?)"
        params.append(crop_filter)

    if search_query:
        query += " AND (LOWER(crop) LIKE ? OR LOWER(market) LIKE ? OR LOWER(state) LIKE ? OR LOWER(district) LIKE ?)"
        wildcard = f"%{search_query.strip().lower()}%"
        params.extend([wildcard, wildcard, wildcard, wildcard])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def delete_prediction(prediction_id: int) -> bool:
    """Deletes a specific prediction entry by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def clear_all_predictions() -> bool:
    """Clears all records from the predictions table."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return True


def get_prediction_count() -> int:
    """Returns the total number of saved predictions."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count

