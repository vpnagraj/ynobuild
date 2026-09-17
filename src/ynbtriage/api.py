"""FastAPI service. This is the ONLY long-running process that opens the SQLite
database. The Streamlit UI talks to it over HTTP and never touches the file.
Run: uvicorn ynbtriage.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import repository as repo
from .db import ensure_schema, get_conn
from .taxonomy import seed_taxonomy


@asynccontextmanager
async def lifespan(_app: FastAPI):
    with get_conn() as conn:
        ensure_schema(conn)
        seed_taxonomy(conn)
    yield


app = FastAPI(title="ynobuild API", version="0.1.0", lifespan=lifespan)


class AnnotationIn(BaseModel):
    class_id: Optional[str] = None
    leaf_id: Optional[str] = None
    status: str = "confirmed"          # confirmed | deferred | skipped
    annotator: str = "unknown"
    note: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/taxonomy")
def taxonomy():
    with get_conn() as conn:
        ensure_schema(conn)
        return repo.get_taxonomy(conn)


@app.get("/stats")
def stats():
    with get_conn() as conn:
        ensure_schema(conn)
        return repo.get_stats(conn)


@app.get("/builds")
def builds(
    class_id: Optional[str] = None,
    leaf_id: Optional[str] = None,
    status: Optional[str] = None,
    split: Optional[str] = None,
    annotated: Optional[str] = Query(None, pattern="^(yes|no|deferred|prefilled)$"),
    q: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    with get_conn() as conn:
        ensure_schema(conn)
        items, total = repo.list_builds(
            conn, class_id=class_id, leaf_id=leaf_id, status=status,
            split=split, annotated=annotated, q=q, limit=limit, offset=offset,
        )
        return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/builds/{build_id}")
def build_detail(build_id: int):
    with get_conn() as conn:
        ensure_schema(conn)
        b = repo.get_build(conn, build_id)
        if not b:
            raise HTTPException(404, "build not found")
        return b


@app.post("/builds/{build_id}/annotations")
def annotate(build_id: int, ann: AnnotationIn):
    with get_conn() as conn:
        ensure_schema(conn)
        if not repo.get_build(conn, build_id):
            raise HTTPException(404, "build not found")
        try:
            return repo.add_annotation(
                conn, build_id,
                class_id=ann.class_id, leaf_id=ann.leaf_id, status=ann.status,
                annotator=ann.annotator, note=ann.note, source="human",
            )
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.delete("/builds/{build_id}")
def delete_build(build_id: int):
    """Permanently delete a build with its annotation history and split row."""
    with get_conn() as conn:
        ensure_schema(conn)
        result = repo.delete_build(conn, build_id)
        if result is None:
            raise HTTPException(404, "build not found")
        return result


@app.post("/maintenance/prune-no-logs")
def prune_no_logs(dry_run: bool = False):
    """Delete every build whose log is the '[no logs]' sentinel. dry_run counts only."""
    with get_conn() as conn:
        ensure_schema(conn)
        return repo.prune_no_logs(conn, dry_run=dry_run)