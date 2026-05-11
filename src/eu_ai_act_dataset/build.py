"""End-to-end build orchestrator.

  fetch(languages) ─► parse_bundle(lang) ─► chunk_act ─► enrich_chunks ─► export

Returns the ExportArtifacts so the caller can publish / inspect / validate.
"""

from __future__ import annotations

import logging
from pathlib import Path

from eu_ai_act_dataset.chunk import Chunk, chunk_act
from eu_ai_act_dataset.config import AI_ACT_CANONICAL_URL, BuildConfig
from eu_ai_act_dataset.enrich import enrich_chunks
from eu_ai_act_dataset.export import ExportArtifacts, export
from eu_ai_act_dataset.fetch import fetch_bundles
from eu_ai_act_dataset.parsers.formex import parse_bundle

log = logging.getLogger(__name__)


def build(
    config: BuildConfig,
    *,
    snapshot_version: str | None = None,
    gold_yaml: Path | None = None,
    template_path: Path | None = None,
) -> ExportArtifacts:
    cache = Path(config.cache_dir)
    output = Path(config.output_dir)

    bundles = fetch_bundles(config.languages, cache)
    log.info("fetched %d bundle(s): %s", len(bundles), [b.language for b in bundles])

    chunks_by_language: dict[str, list[Chunk]] = {}
    for bundle in bundles:
        act = parse_bundle(
            bundle.directory,
            document_id="reg_2024_1689",
            language=bundle.language,
            source_url=AI_ACT_CANONICAL_URL,
        )
        chunks = chunk_act(act, source_type="regulation")
        enrich_chunks(chunks)

        # Apply config filters.
        if not config.include_article_full:
            chunks = [c for c in chunks if c.chunk_type != "article_full"]
        if not config.include_recitals:
            chunks = [c for c in chunks if c.chunk_type != "recital"]
        if not config.include_annexes:
            chunks = [c for c in chunks if c.chunk_type != "annex_item"]

        log.info(
            "lang=%s: %d recitals, %d articles, %d annexes → %d chunks",
            bundle.language, len(act.recitals), len(act.articles), len(act.annexes), len(chunks),
        )
        chunks_by_language[bundle.language] = chunks

    return export(
        chunks_by_language,
        bundles,
        config,
        output_dir=output,
        snapshot_version=snapshot_version,
        gold_yaml=gold_yaml,
        template_path=template_path,
    )
