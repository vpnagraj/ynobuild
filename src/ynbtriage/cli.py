"""`ynbtriage` command-line entry point. Each batch container runs one subcommand.

Every command calls ensure_schema + seed_taxonomy first, so any of them is safe
to run against a fresh, empty volume regardless of ordering.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from . import fetch as fetch_mod
from . import ingest as ingest_mod
from . import repository as repo
from . import seed as seed_mod
from . import splits as splits_mod
from .db import ensure_schema, get_conn
from .repository import get_stats
from .taxonomy import seed_taxonomy

app = typer.Typer(help="ynobuild — build-failure triage corpus / DB tooling", no_args_is_help=True)


@app.command()
def initdb():
    """Create the schema and seed the taxonomy (idempotent)."""
    with get_conn() as conn:
        ensure_schema(conn)
        seed_taxonomy(conn)
    typer.echo("schema + taxonomy ready")


@app.command("load-csv")
def load_csv(csv_path: Path = typer.Argument(..., exists=True, readable=True)):
    """Ingest a CSV of build logs (optionally with a predefined leaf label) into the builds table."""
    ins, upd, skip, labeled = ingest_mod.load_csv(str(csv_path))
    typer.echo(f"inserted {ins}, updated {upd}, skipped {skip}, labeled {labeled}")


@app.command("fetch-dockerfiles")
def fetch_dockerfiles(
    limit: int = typer.Option(None, help="Only process this many builds."),
    sleep: float = typer.Option(0.5, help="Seconds between requests (be kind to forges)."),
):
    """Fetch Dockerfiles from GitHub/GitLab for builds that don't have one."""
    ok, miss, err = fetch_mod.fetch_dockerfiles(limit=limit, sleep=sleep)
    typer.echo(f"ok {ok}, not_found {miss}, error {err}")


@app.command("assign-splits")
def assign_splits(
    seed: int = 7400,
    gold_frac: float = 0.20,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Assign stratified train/val/test/gold splits over labelled builds."""
    counts = splits_mod.assign_splits(
        seed=seed, gold_frac=gold_frac, val_frac=val_frac, test_frac=test_frac
    )
    typer.echo(f"splits: {counts}")


@app.command("demo-seed")
def demo_seed(n: int = 12):
    """Insert a small set of synthetic, representative builds to try the UI."""
    c = seed_mod.demo_seed(n)
    typer.echo(f"inserted {c} demo builds")


@app.command()
def stats():
    """Print corpus statistics as JSON."""
    with get_conn() as conn:
        ensure_schema(conn)
        seed_taxonomy(conn)
        typer.echo(json.dumps(get_stats(conn), indent=2))


@app.command("delete-build")
def delete_build(
    build_id: int = typer.Argument(None, help="Numeric build_id to delete."),
    external_id: str = typer.Option(None, "--external-id", help="Delete by external_id instead."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Permanently delete one build and its annotation history."""
    if (build_id is None) == (external_id is None):
        raise typer.BadParameter("Provide exactly one of BUILD_ID or --external-id.")
    with get_conn() as conn:
        ensure_schema(conn)
        bid = build_id if build_id is not None else repo.find_build_id(conn, external_id)
        if bid is None:
            typer.echo("no matching build")
            raise typer.Exit(1)
        if not yes and not typer.confirm(f"Delete build {bid} and its annotations?"):
            typer.echo("aborted")
            raise typer.Exit(1)
        res = repo.delete_build(conn, bid)
        typer.echo(
            f"deleted build {res['build_id']} "
            f"({res['annotations_deleted']} annotations, {res['split_deleted']} split rows)"
        )


@app.command("prune-no-logs")
def prune_no_logs(
    dry_run: bool = typer.Option(False, "--dry-run", help="Only count; delete nothing."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Delete every build whose log is the '[no logs]' sentinel."""
    with get_conn() as conn:
        ensure_schema(conn)
        n = repo.prune_no_logs(conn, dry_run=True)["count"]
        if dry_run:
            typer.echo(f"{n} build(s) with no logs (dry run; nothing deleted)")
            return
        if n == 0:
            typer.echo("nothing to prune")
            return
        if not yes and not typer.confirm(f"Delete {n} build(s) with no logs?"):
            typer.echo("aborted")
            raise typer.Exit(1)
        res = repo.prune_no_logs(conn, dry_run=False)
        typer.echo(f"deleted {res['count']} build(s)")


def main():
    app()


if __name__ == "__main__":
    main()