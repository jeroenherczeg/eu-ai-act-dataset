"""eu-ai-act-dataset — structured chunks of Regulation (EU) 2024/1689 for retrieval research.

The pipeline:
    fetch ─► parse ─► chunk ─► enrich ─► export ─► publish

`build` orchestrates fetch → enrich → export.
`publish` pushes the exported parquet bundle to a Hugging Face dataset repo.
Each stage is independently testable; the CLI in `cli.py` glues them together.
"""

__version__ = "0.1.0"
