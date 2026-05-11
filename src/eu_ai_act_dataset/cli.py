"""Typer CLI:

  eu-ai-act-dataset build         fetch → parse → chunk → enrich → export to dist/
  eu-ai-act-dataset validate dist/
  eu-ai-act-dataset diff   dist/ --repo USER/REPO
  eu-ai-act-dataset publish dist/ --repo USER/REPO --version vYYYY-MM-DD
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(override=False)

from eu_ai_act_dataset.build import build as run_build
from eu_ai_act_dataset.config import BuildConfig
from eu_ai_act_dataset.diff import diff_against_hf
from eu_ai_act_dataset.publish import publish as run_publish
from eu_ai_act_dataset.validate import validate as run_validate

app = typer.Typer(no_args_is_help=True, add_completion=False, help="EU AI Act dataset builder")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("build")
def build(
    output: Annotated[Path, typer.Option("--output", "-o", help="output directory")] = Path("dist"),
    cache: Annotated[Path, typer.Option("--cache", help="raw-bundle cache dir")] = Path(".dataset_cache/raw"),
    languages: Annotated[
        list[str], typer.Option("--lang", "-l", help="languages to build (repeatable). Defaults to en,nl,fr")
    ] = None,
    snapshot: Annotated[str | None, typer.Option("--snapshot", help="snapshot version, e.g. v2024-10-17")] = None,
    gold: Annotated[Path | None, typer.Option("--gold", help="path to gold/retrieval_eval.yaml")] = Path("gold/retrieval_eval.yaml"),
    no_recitals: Annotated[bool, typer.Option("--no-recitals")] = False,
    no_annexes: Annotated[bool, typer.Option("--no-annexes")] = False,
    no_article_full: Annotated[bool, typer.Option("--no-article-full")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch, parse, chunk, enrich and export the dataset to `--output`."""
    _setup_logging(verbose)
    cfg = BuildConfig(
        languages=languages or ["en", "nl", "fr"],
        output_dir=str(output),
        cache_dir=str(cache),
        include_recitals=not no_recitals,
        include_annexes=not no_annexes,
        include_article_full=not no_article_full,
    )
    artifacts = run_build(cfg, snapshot_version=snapshot, gold_yaml=gold)
    t = Table(title="build summary")
    t.add_column("field")
    t.add_column("value")
    t.add_row("snapshot", artifacts.snapshot_version)
    t.add_row("rows", str(artifacts.row_count))
    t.add_row("parquet", str(artifacts.chunks_parquet))
    t.add_row("card", str(artifacts.dataset_card))
    t.add_row("sources.csv", str(artifacts.sources_csv))
    t.add_row("gold", str(artifacts.gold_parquet) if artifacts.gold_parquet else "—")
    console.print(t)


@app.command("validate")
def validate(
    dist: Annotated[Path, typer.Argument()] = Path("dist"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Sanity-check a built dist/. Exits non-zero on any failure."""
    _setup_logging(verbose)
    failures = run_validate(dist)
    if failures:
        console.print(f"[red]✗[/red] {failures} validation failures")
        sys.exit(1)
    console.print("[green]✓[/green] validation passed")


@app.command("diff")
def diff(
    dist: Annotated[Path, typer.Argument()] = Path("dist"),
    repo: Annotated[str, typer.Option("--repo", help="HF dataset repo id")] = "jeroenherczeg/eu-ai-act",
    emit_summary: Annotated[Path | None, typer.Option("--emit-summary")] = None,
) -> None:
    """Diff dist/ against the HF dataset's current state."""
    summary = diff_against_hf(dist, repo)
    console.print(summary)
    if emit_summary:
        with emit_summary.open("a") as fh:
            fh.write(summary)


@app.command("publish")
def publish(
    dist: Annotated[Path, typer.Argument()] = Path("dist"),
    repo: Annotated[str, typer.Option("--repo")] = "jeroenherczeg/eu-ai-act",
    version: Annotated[str | None, typer.Option("--version", help="overrides dataset_info.json")] = None,
    private: Annotated[bool, typer.Option("--private")] = False,
) -> None:
    """Push dist/ to a Hugging Face dataset repo. Requires HF_TOKEN env."""
    import json
    info = json.loads((dist / "dataset_info.json").read_text())
    snap = version or info["version"]
    url = run_publish(dist, repo=repo, snapshot_version=snap, private=private)
    console.print(f"[green]✓[/green] published {snap} → {url}")


if __name__ == "__main__":
    app()
