"""
SQLite detection history — stores cat detections with timestamp.
"""
import sqlite3
import time
from pathlib import Path

import os
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "reports/detections.db"))


class DetectionDB:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    label      TEXT    NOT NULL,
                    confidence REAL    NOT NULL,
                    source     TEXT    NOT NULL,
                    timestamp  REAL    NOT NULL
                )
            """)

    def save(self, label: str, confidence: float, source: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO detections (label, confidence, source, timestamp) VALUES (?,?,?,?)",
                (label, round(confidence * 100, 1), source, time.time())
            )

    def get_recent(self, limit: int = 20):
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT label, confidence, source, timestamp
                   FROM detections
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()

        return [
            {
                "label"      : r[0],
                "confidence" : r[1],
                "source"     : r[2],
                "time"       : _fmt_time(r[3]),
            }
            for r in rows
        ]


def _fmt_time(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")