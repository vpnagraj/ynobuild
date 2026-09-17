"""SQLite connection management.

WAL mode lets readers proceed during a write; `busy_timeout` makes a would-be
concurrent writer wait instead of failing immediately. That combination is what
makes the single-writer discipline (only the API writes online; ingest/fetch/
splits are offline batch jobs) safe enough for this project.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")  # 30s
    return conn


@contextmanager
def get_conn(db_path: Path | str | None = None):
    """Transaction-scoped connection: commit on success, rollback on error."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: safe to call on every process start."""
    conn.executescript(SCHEMA_PATH.read_text())
