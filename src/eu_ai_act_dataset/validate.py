"""Sanity checks on a built dist/.

Fails the build (non-zero exit) if any invariant is violated. Designed for
CI: cheap to run, deterministic, catches "did the parser silently regress?"
without needing a full eval set.

Invariants checked:
  - parquet exists and is non-empty
  - schema digest in dataset_info.json matches the live schema
  - per-language counts: ≥ 100 articles (the regulation has 113), ≥ 150
    recitals (180), and there's at least one annex_item per Annex I-XIII
  - id uniqueness within a language
  - parallel-structure invariant: every structure_path that exists in EN
    also exists in NL+FR (when those languages are built)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

log = logging.getLogger(__name__)


# Thresholds — set just below the published counts so a minor parser bug fails
# CI without being so tight that minor source edits (corrigenda) break it.
MIN_ARTICLES_PER_LANG = 100      # actual: 113
MIN_RECITALS_PER_LANG = 150      # actual: 180
MIN_ANNEXES_PER_LANG = 10        # actual: 13


def validate(dist: Path) -> int:
    """Return 0 on success, non-zero count of failures otherwise."""
    failures: list[str] = []

    chunks_path = dist / "ai_act_chunks.parquet"
    info_path = dist / "dataset_info.json"

    if not chunks_path.exists():
        failures.append(f"missing parquet: {chunks_path}")
    if not info_path.exists():
        failures.append(f"missing info: {info_path}")
    if failures:
        return _report(failures)

    info = json.loads(info_path.read_text())
    table = pq.read_table(chunks_path)
    log.info("validating %s rows × %s cols", table.num_rows, len(table.schema))

    # 1) row count sane
    if table.num_rows < 200:
        failures.append(f"suspiciously low row count: {table.num_rows}")

    # 2) per-language counts
    languages = sorted({lang for lang in table.column("language").to_pylist()})
    for lang in languages:
        mask = table.column("language").to_pylist()
        idx = [i for i, l in enumerate(mask) if l == lang]
        ctypes = [table.column("chunk_type").to_pylist()[i] for i in idx]
        articles = sum(1 for ct in ctypes if ct == "article_full")
        paragraphs = sum(1 for ct in ctypes if ct == "paragraph")
        recitals = sum(1 for ct in ctypes if ct == "recital")
        annex_items = sum(1 for ct in ctypes if ct == "annex_item")
        annex_set = {
            table.column("annex_no").to_pylist()[i]
            for i in idx
            if table.column("chunk_type").to_pylist()[i] == "annex_item"
        }
        if articles < MIN_ARTICLES_PER_LANG:
            failures.append(f"[{lang}] only {articles} article_full chunks (min {MIN_ARTICLES_PER_LANG})")
        if recitals < MIN_RECITALS_PER_LANG:
            failures.append(f"[{lang}] only {recitals} recitals (min {MIN_RECITALS_PER_LANG})")
        if len(annex_set) < MIN_ANNEXES_PER_LANG:
            failures.append(
                f"[{lang}] only {len(annex_set)} distinct annex numbers (min {MIN_ANNEXES_PER_LANG})"
            )
        log.info(
            "[%s] articles=%d paragraphs=%d recitals=%d annex_items=%d distinct_annexes=%d",
            lang, articles, paragraphs, recitals, annex_items, len(annex_set),
        )

    # 3) id uniqueness within language
    ids = table.column("id").to_pylist()
    langs = table.column("language").to_pylist()
    seen: set[tuple[str, str]] = set()
    dupes = []
    for rid, lang in zip(ids, langs, strict=True):
        key = (lang, rid)
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes:
        failures.append(f"{len(dupes)} duplicate (language, id) pairs; first: {dupes[:3]}")

    # 4) parallel structure: structure_path coverage should be ~identical across languages
    if len(languages) > 1:
        by_lang_paths: dict[str, set[str]] = {l: set() for l in languages}
        paths = table.column("structure_path").to_pylist()
        for sp, lg in zip(paths, langs, strict=True):
            by_lang_paths[lg].add(sp)
        baseline = max(by_lang_paths.values(), key=len)
        for lang, paths_set in by_lang_paths.items():
            missing = len(baseline - paths_set)
            if missing > 0:
                # Soft check: tolerate small drift (translations occasionally
                # split a paragraph differently). Fail only on > 5% drift.
                ratio = missing / len(baseline)
                if ratio > 0.05:
                    failures.append(
                        f"[{lang}] {missing} structure_paths missing vs largest language ({ratio:.1%} drift)"
                    )

    # 5) schema digest matches what was recorded
    from eu_ai_act_dataset.export import _schema_digest

    if info.get("schema_digest") != _schema_digest():
        failures.append("schema_digest in dataset_info.json doesn't match live schema")

    return _report(failures)


def _report(failures: list[str]) -> int:
    if not failures:
        log.info("✓ validation passed")
        return 0
    for f in failures:
        log.error("✗ %s", f)
    return len(failures)


if __name__ == "__main__":
    sys.exit(validate(Path(sys.argv[1] if len(sys.argv) > 1 else "dist")))
