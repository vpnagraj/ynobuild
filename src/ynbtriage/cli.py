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


# ---- modeling (needs the [model] extra; imported lazily so the core CLI/API stay light)

@app.command()
def train(
    out: Path = typer.Option(..., "--out", help="Artifact directory to write (e.g. models/run-001)."),
    db: Path = typer.Option(None, "--db", help="DB path (default: YNB_DB_PATH)."),
    target: str = typer.Option("class", help="class (3 coarse) | leaf (12 leaves)."),
    features: str = typer.Option("embed", help="embed | tfidf | both."),
    include_prefilled: bool = typer.Option(False, help="Add prefilled (silver) labels to TRAIN only."),
    hidden: str = typer.Option("64", help="Hidden layer sizes, comma-separated, e.g. 128,64."),
    activation: str = typer.Option("relu", help="relu | tanh | gelu | sigmoid."),
    dropout: float = 0.2,
    optimizer: str = typer.Option("adam", help="adam | sgd."),
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = typer.Option(25, help="Early-stopping patience on val loss (0 = off)."),
    seed: int = 7400,
    quiet: bool = typer.Option(False, help="Don't print per-epoch metrics."),
):
    """Train baselines + the MLP on confirmed labels; write weights, metrics and the loss curve."""
    from .model.pipeline import run
    from .model.train import TrainConfig

    cfg = TrainConfig(hidden=tuple(int(h) for h in hidden.split(",") if h), activation=activation,
                      dropout=dropout, optimizer=optimizer, lr=lr, epochs=epochs,
                      batch_size=batch_size, patience=patience, seed=seed)

    def show(rec):
        if not quiet and (rec["epoch"] == 1 or rec["epoch"] % 10 == 0):
            typer.echo(f"epoch {rec['epoch']:4d}  train_loss {rec['train_loss']:.4f}  "
                       f"val_loss {rec['val_loss']:.4f}  train_acc {rec['train_acc']:.3f}  "
                       f"val_acc {rec['val_acc']:.3f}")

    s = run(out, db_path=db, target=target, features=features,
            include_prefilled=include_prefilled, cfg=cfg, on_epoch=show)

    typer.echo(f"\n{s['n']}  epochs run {s['epochs_run']}, best epoch {s['best_epoch']}")
    typer.echo(f"leakage: {s['leakage']}")
    typer.echo(f"\n{'model':14s} {'val acc':>8s} {'val F1':>8s} {'test acc':>9s} {'test F1':>8s}")
    rows = {**s["baselines"], "mlp": s["mlp"]}
    for name, r in rows.items():
        typer.echo(f"{name:14s} {r['val']['accuracy']:8.3f} {r['val']['macro_f1']:8.3f} "
                   f"{r['test']['accuracy']:9.3f} {r['test']['macro_f1']:8.3f}")
    typer.echo(f"\nwrote {out}/ (model.pt, featurizer.joblib, metrics.json, history.csv, loss_curve.png)")


@app.command()
def predict(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    build_id: int = typer.Option(None, "--build-id", help="Classify this build from the DB."),
    log_file: Path = typer.Option(None, "--log-file", exists=True, help="Or classify a log file."),
    db: Path = typer.Option(None, "--db"),
):
    """Classify one build's log with a trained model."""
    from .model.predict import Predictor

    if (build_id is None) == (log_file is None):
        raise typer.BadParameter("Provide exactly one of --build-id or --log-file.")
    if log_file:
        log = log_file.read_text(errors="replace")
    else:
        with get_conn(db) as conn:
            b = repo.get_build(conn, build_id)
        if not b:
            typer.echo("no such build")
            raise typer.Exit(1)
        log = b["log_tail"]
    typer.echo(json.dumps(Predictor.load(model_dir).predict(log), indent=2))


@app.command()
def baselines(
    out: Path = typer.Option(..., "--out", help="Run directory to write baselines.json into (e.g. models/final)."),
    db: Path = typer.Option(None, "--db", help="DB path (default: YNB_DB_PATH)."),
    target: str = typer.Option("class", help="class | leaf (must match the run's target)."),
    seed: int = 7400,
):
    """Score only the shallow baselines and write baselines.json next to a run (no retraining)."""
    from .model.pipeline import run_baselines

    r = run_baselines(out, db_path=db, target=target, seed=seed)
    typer.echo(f"{r['n']}  data fingerprint {r['data_fingerprint']['hash']}")

    mpath = out / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        run_fp = (m.get("data_fingerprint") or {}).get("hash")
        if run_fp is None:
            typer.echo("note: this run predates data fingerprints, so I can't confirm it was trained "
                       f"on the same rows. Its split sizes were {m.get('n')}.")
        elif run_fp != r["data_fingerprint"]["hash"]:
            typer.echo(f"WARNING: this run was trained on different data (fingerprint {run_fp}); "
                       "these baseline scores are not directly comparable with it.")
        if m.get("target") and m["target"] != target:
            typer.echo(f"WARNING: the run's target is {m['target']!r}, not {target!r}.")

    typer.echo(f"\n{'model':14s} {'val acc':>8s} {'val F1':>8s} {'test acc':>9s} {'test F1':>8s}")
    for name, s in r["baselines"].items():
        typer.echo(f"{name:14s} {s['val']['accuracy']:8.3f} {s['val']['macro_f1']:8.3f} "
                   f"{s['test']['accuracy']:9.3f} {s['test']['macro_f1']:8.3f}")
    typer.echo(f"\nwrote {out}/baselines.json")


@app.command("shrink-model")
def shrink_model(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    db: Path = typer.Option(None, "--db", help="Check predictions on every build in this DB "
                                                "(default: on 300 synthetic logs)."),
):
    """Shrink featurizer.joblib (float32 projection + compression) without changing predictions.

    Predictions from the original and the shrunk featurizer are compared first;
    if any predicted label differs, nothing is written.
    """
    import joblib
    import numpy as np

    from .model.predict import Predictor

    fpath = model_dir / "featurizer.joblib"
    before = fpath.stat().st_size
    original = Predictor.load(model_dir)
    shrunk = joblib.load(fpath)
    for part in getattr(shrunk, "parts", [shrunk]):
        if hasattr(part, "svd"):
            part.svd.components_ = part.svd.components_.astype(np.float32)

    if db:
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            logs = [r[0] for r in conn.execute("SELECT log_tail FROM builds")]
        source = f"{len(logs)} builds in {db}"
    else:
        import random
        from .model.synth import TEMPLATES, make_log
        rnd = random.Random(0)
        logs = [make_log(rnd.choice(list(TEMPLATES)), rnd) for _ in range(300)]
        source = "300 synthetic logs (pass --db to check on your real builds)"

    a = original.predict_many(logs)
    b = Predictor(original.model, shrunk, original.labels, original.target).predict_many(logs)
    flips = sum(x["label"] != y["label"] for x, y in zip(a, b))
    diff = max(abs(x["probabilities"][l] - y["probabilities"][l])
               for x, y in zip(a, b) for l in original.labels)
    typer.echo(f"checked {source}: label changes {flips}, max probability change {diff:.1e}")
    if flips:
        typer.echo("predictions changed; featurizer.joblib left as it was")
        raise typer.Exit(1)

    tmp = fpath.with_name("featurizer.joblib.tmp")
    joblib.dump(shrunk, tmp, compress=3)
    tmp.replace(fpath)
    typer.echo(f"featurizer.joblib: {before / 1e6:.1f} MB -> {fpath.stat().st_size / 1e6:.1f} MB")


@app.command("synth-db")
def synth_db(
    db: Path = typer.Option(..., "--db", help="A SEPARATE DB file for synthetic data."),
    n: int = 600,
    seed: int = 7400,
    label_noise: float = 0.05,
):
    """Create a synthetic labelled DB for developing the model before annotation is done."""
    from .model.synth import synth_db as _synth

    counts = _synth(db, n=n, seed=seed, label_noise=label_noise)
    typer.echo(f"synthetic builds written to {db}; splits: {counts}")


def main():
    app()


if __name__ == "__main__":
    main()