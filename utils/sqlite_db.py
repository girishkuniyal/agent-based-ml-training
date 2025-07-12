# utils/sqlite_db.py
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("db/app_state.sqlite")

# Ensure DB directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS usecases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        path TEXT,
        port INTEGER,
        status TEXT,
        api_hits INTEGER DEFAULT 0,
        model_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()


def register_usecase(name: str, path: str, port: int, model_type: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO usecases (name, path, port, status, model_type)
        VALUES (?, ?, ?, 'running', ?)
    """, (name, path, port, model_type))
    conn.commit()
    conn.close()


def update_status(name: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        UPDATE usecases SET status=? WHERE name=?
    """, (status, name))
    conn.commit()
    conn.close()


def increment_hit(name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        UPDATE usecases SET api_hits = api_hits + 1 WHERE name=?
    """, (name,))
    conn.commit()
    conn.close()


def get_all_usecases():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, port, status, api_hits, model_type FROM usecases")
    results = cur.fetchall()
    conn.close()
    return results


def get_usecase(name: str) -> Optional[tuple]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM usecases WHERE name=?", (name,))
    result = cur.fetchone()
    conn.close()
    return result
