"""Export enriched chunks to a Hugging Face-loadable parquet bundle.

Output layout (relative to dist/):

    dist/
    ├── ai_act_chunks.parquet     # all languages, one row per chunk
    ├── sources.csv               # per-language provenance (URL, sha256, retrieved_at)
    ├── dataset_info.json         # row counts, schema digest, version tag
    ├── README.md                 # rendered dataset card (HF reads this for display)
    └── gold/
        └── retrieval_eval.parquet (optional, only if gold/retrieval_eval.yaml has entries)

The parquet is loadable via:

    from datasets import load_dataset
    ds = load_dataset("<user>/eu-ai-act-2024-1689", split="train")
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from eu_ai_act_dataset.chunk import Chunk
from eu_ai_act_dataset.config import (
    AI_ACT_CANONICAL_URL,
    AI_ACT_CELEX,
    AI_ACT_ELI,
    BuildConfig,
)
from eu_ai_act_dataset.fetch import FetchedBundle

log = logging.getLogger(__name__)


@dataclass
class ExportArtifacts:
    output_dir: Path
    chunks_parquet: Path
    sources_csv: Path
    dataset_info: Path
    dataset_card: Path
    gold_parquet: Path | None
    snapshot_version: str
    row_count: int


def export(
    chunks_by_language: dict[str, list[Chunk]],
    bundles: list[FetchedBundle],
    config: BuildConfig,
    *,
    output_dir: Path,
    snapshot_version: str | None = None,
    gold_yaml: Path | None = None,
    template_path: Path | None = None,
) -> ExportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gold").mkdir(parents=True, exist_ok=True)

    if snapshot_version is None:
        # Derive from the most recent bundle. The Cellar mtime would be more
        # authoritative, but the build-time date is a fine fallback and easy
        # to reason about.
        snapshot_version = "v" + date.today().isoformat()

    # 1) chunks parquet — every chunk row across all languages
    table = _rows_to_arrow(chunks_by_language, snapshot_version, bundles)
    parquet_path = output_dir / "ai_act_chunks.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    row_count = table.num_rows
    log.info("wrote %s rows to %s", row_count, parquet_path)

    # 2) sources.csv — provenance per language
    sources_csv = output_dir / "sources.csv"
    _write_sources_csv(sources_csv, bundles, snapshot_version)

    # 3) gold/retrieval_eval.parquet — optional retrieval eval split
    gold_parquet = _maybe_write_gold(output_dir, gold_yaml)

    # 4) dataset_info.json
    info_path = output_dir / "dataset_info.json"
    info = _build_dataset_info(
        snapshot_version=snapshot_version,
        languages=list(chunks_by_language.keys()),
        row_count=row_count,
        chunks_by_language=chunks_by_language,
        bundles=bundles,
        gold_present=gold_parquet is not None,
    )
    info_path.write_text(json.dumps(info, indent=2, default=_json_default))

    # 5) README.md (the HF dataset card)
    card_path = output_dir / "README.md"
    template = (template_path or _default_template()).read_text()
    card_path.write_text(_render_card(template, info))

    return ExportArtifacts(
        output_dir=output_dir,
        chunks_parquet=parquet_path,
        sources_csv=sources_csv,
        dataset_info=info_path,
        dataset_card=card_path,
        gold_parquet=gold_parquet,
        snapshot_version=snapshot_version,
        row_count=row_count,
    )


# --- arrow / parquet --------------------------------------------------------


_PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("language", pa.string()),
        ("text", pa.string()),
        ("chunk_type", pa.string()),
        ("citation_label", pa.string()),
        ("structure_path", pa.string()),
        ("title_no", pa.int32()),
        ("chapter_no", pa.int32()),
        ("section_no", pa.int32()),
        ("article_no", pa.int32()),
        ("paragraph_no", pa.int32()),
        ("recital_no", pa.int32()),
        ("annex_no", pa.string()),
        ("annex_section", pa.string()),
        ("references_articles", pa.list_(pa.int32())),
        ("interprets_articles", pa.list_(pa.int32())),
        ("defined_terms", pa.list_(pa.string())),
        ("effective_from", pa.date32()),
        ("transitional", pa.bool_()),
        ("document_id", pa.string()),
        ("celex", pa.string()),
        ("source_url", pa.string()),
        ("source_publisher", pa.string()),
        ("license", pa.string()),
        ("snapshot_version", pa.string()),
        ("bundle_sha256", pa.string()),
        ("retrieved_at", pa.date32()),
        ("parent_structure_path", pa.string()),
    ]
)


def _rows_to_arrow(
    chunks_by_language: dict[str, list[Chunk]],
    snapshot_version: str,
    bundles: list[FetchedBundle],
) -> pa.Table:
    sha_by_lang = {b.language: b.bundle_sha256 for b in bundles}
    today = date.today()
    rows: dict[str, list] = {field.name: [] for field in _PARQUET_SCHEMA}
    for language, chunks in chunks_by_language.items():
        sha = sha_by_lang.get(language, "")
        for c in chunks:
            structure = c.structure or {}
            rows["id"].append(_row_id(c, language))
            rows["language"].append(language)
            rows["text"].append(c.text)
            rows["chunk_type"].append(c.chunk_type)
            rows["citation_label"].append(c.citation_label)
            rows["structure_path"].append(c.structure_path)
            rows["title_no"].append(structure.get("title_no"))
            rows["chapter_no"].append(structure.get("chapter_no"))
            rows["section_no"].append(structure.get("section_no"))
            rows["article_no"].append(structure.get("article_no"))
            rows["paragraph_no"].append(structure.get("paragraph_no"))
            rows["recital_no"].append(structure.get("recital_no"))
            rows["annex_no"].append(structure.get("annex_no"))
            rows["annex_section"].append(structure.get("annex_section"))
            rows["references_articles"].append(c.references_articles or [])
            rows["interprets_articles"].append(c.interprets_articles or [])
            rows["defined_terms"].append(c.defined_terms or [])
            rows["effective_from"].append(c.effective_from)
            rows["transitional"].append(bool(c.transitional))
            rows["document_id"].append(c.document_id)
            rows["celex"].append(AI_ACT_CELEX)
            rows["source_url"].append(c.source_url or AI_ACT_CANONICAL_URL)
            rows["source_publisher"].append("European Union")
            rows["license"].append("CC BY 4.0")
            rows["snapshot_version"].append(snapshot_version)
            rows["bundle_sha256"].append(sha)
            rows["retrieved_at"].append(today)
            rows["parent_structure_path"].append(c.parent_structure_path)
    return pa.table(rows, schema=_PARQUET_SCHEMA)


def _row_id(c: Chunk, language: str) -> str:
    """Stable, human-readable id like 'art-9-par-2-en' or 'rec-102-en'.

    The id is unique within a (document_id, language) and is intentionally
    derivable from the structure_path so two languages of the same provision
    share an id prefix up to the language suffix.
    """
    path_slug = c.structure_path.replace(":", "-").replace("/", "-")
    return f"{path_slug}-{language}"


# --- sources.csv ------------------------------------------------------------


def _write_sources_csv(path: Path, bundles: list[FetchedBundle], snapshot_version: str) -> None:
    today = date.today().isoformat()
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "language",
                "celex",
                "eli",
                "canonical_url",
                "formex_bundle_url",
                "bundle_sha256",
                "retrieved_at",
                "snapshot_version",
                "license",
                "license_notice_url",
            ]
        )
        for b in bundles:
            w.writerow(
                [
                    b.language,
                    AI_ACT_CELEX,
                    AI_ACT_ELI,
                    AI_ACT_CANONICAL_URL,
                    f"http://publications.europa.eu/resource/oj/L_202401689.{b.iso3}.fmx4",
                    b.bundle_sha256,
                    today,
                    snapshot_version,
                    "CC BY 4.0",
                    "https://eur-lex.europa.eu/content/legal-notice/legal-notice.html",
                ]
            )


# --- gold split -------------------------------------------------------------


_GOLD_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("question", pa.string()),
        ("category", pa.string()),
        ("difficulty", pa.string()),
        ("required_citations", pa.list_(pa.string())),
        ("expected_must_mention", pa.list_(pa.string())),
        ("must_not_mention", pa.list_(pa.string())),
        ("retrieval_must_include", pa.list_(pa.string())),
    ]
)


def _maybe_write_gold(output_dir: Path, gold_yaml: Path | None) -> Path | None:
    if gold_yaml is None or not gold_yaml.exists():
        return None
    items = yaml.safe_load(gold_yaml.read_text()) or []
    if not items:
        return None
    rows: dict[str, list] = {f.name: [] for f in _GOLD_SCHEMA}
    for it in items:
        rows["id"].append(it.get("id") or "")
        rows["question"].append(it.get("question") or "")
        rows["category"].append(it.get("category") or "")
        rows["difficulty"].append(it.get("difficulty") or "")
        rows["required_citations"].append(it.get("required_citations") or [])
        rows["expected_must_mention"].append(it.get("expected_answer_must_mention") or [])
        rows["must_not_mention"].append(it.get("must_not_mention") or [])
        rows["retrieval_must_include"].append(it.get("retrieval_must_include") or [])
    out = output_dir / "gold" / "retrieval_eval.parquet"
    pq.write_table(pa.table(rows, schema=_GOLD_SCHEMA), out, compression="zstd")
    log.info("wrote %d gold-set rows to %s", len(items), out)
    return out


# --- dataset_info.json + dataset card --------------------------------------


def _build_dataset_info(
    *,
    snapshot_version: str,
    languages: list[str],
    row_count: int,
    chunks_by_language: dict[str, list[Chunk]],
    bundles: list[FetchedBundle],
    gold_present: bool,
) -> dict:
    by_lang_breakdown = {
        lang: {
            "total": len(chunks),
            "by_type": _count_by_type(chunks),
        }
        for lang, chunks in chunks_by_language.items()
    }
    return {
        "name": "eu-ai-act-2024-1689",
        "version": snapshot_version,
        "celex": AI_ACT_CELEX,
        "eli": AI_ACT_ELI,
        "canonical_url": AI_ACT_CANONICAL_URL,
        "languages": languages,
        "rows": row_count,
        "by_language": by_lang_breakdown,
        "license": "CC BY 4.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "schema_digest": _schema_digest(),
        "gold_split": gold_present,
        "bundles": [
            {"language": b.language, "iso3": b.iso3, "sha256": b.bundle_sha256}
            for b in bundles
        ],
    }


def _count_by_type(chunks: list[Chunk]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in chunks:
        out[c.chunk_type] = out.get(c.chunk_type, 0) + 1
    return out


def _schema_digest() -> str:
    """Stable hash of the parquet schema. Lets consumers detect breaking changes."""
    blob = "|".join(f"{f.name}:{f.type}" for f in _PARQUET_SCHEMA).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _render_card(template: str, info: dict) -> str:
    """Tiny template renderer — fills `{{ key }}` style placeholders.

    Avoids pulling in Jinja for a half-dozen placeholders.
    """
    # Pre-build derived strings
    languages_str = ", ".join(info["languages"])
    by_type_total: dict[str, int] = {}
    for breakdown in info["by_language"].values():
        for k, v in breakdown["by_type"].items():
            by_type_total[k] = by_type_total.get(k, 0) + v

    by_type_md = "\n".join(
        f"- `{ct}`: **{n}**" for ct, n in sorted(by_type_total.items())
    )
    by_lang_md = "\n".join(
        f"- **{lang}**: {breakdown['total']} chunks"
        for lang, breakdown in sorted(info["by_language"].items())
    )

    substitutions = {
        "{{ version }}": info["version"],
        "{{ celex }}": info["celex"],
        "{{ eli }}": info["eli"],
        "{{ canonical_url }}": info["canonical_url"],
        "{{ languages }}": languages_str,
        "{{ language_yaml }}": "\n".join(f"  - {lang}" for lang in info["languages"]),
        "{{ rows }}": str(info["rows"]),
        "{{ schema_digest }}": info["schema_digest"],
        "{{ built_at }}": info["built_at"],
        "{{ chunk_type_breakdown }}": by_type_md,
        "{{ language_breakdown }}": by_lang_md,
        "{{ gold_section }}": _GOLD_SECTION if info["gold_split"] else _NO_GOLD_SECTION,
    }
    out = template
    for k, v in substitutions.items():
        out = out.replace(k, v)
    return out


_GOLD_SECTION = """\
## Gold retrieval split

A small `gold/retrieval_eval.parquet` config carries hand-curated question / required-citation pairs.
Each item names the chunk `structure_path`s a correct retrieval must return, enabling recall@k and
citation-correctness benchmarks against this corpus.

```python
from datasets import load_dataset
gold = load_dataset("PATH/TO/REPO", data_files="gold/retrieval_eval.parquet")
```
"""

_NO_GOLD_SECTION = ""


def _default_template() -> Path:
    return Path(__file__).resolve().parents[2] / "DATASET_CARD.md.tmpl"


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"can't serialize {type(o).__name__}")
