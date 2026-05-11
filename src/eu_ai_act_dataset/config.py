"""Build configuration.

A single source for v1 (the regulation itself, Formex from Cellar). The
languages list is the one knob that materially changes the output shape:
publishing EN+NL+FR yields a parallel multilingual corpus where rows for the
same provision share an `id` prefix and a `structure_path`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# CELEX → Cellar mapping for Regulation (EU) 2024/1689.
AI_ACT_CELEX = "32024R1689"
AI_ACT_CELLAR_ID = "dc8116a1-3fe6-11ef-865a-01aa75ed71a1"
AI_ACT_ELI = "http://data.europa.eu/eli/reg/2024/1689/oj"
AI_ACT_CANONICAL_URL = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj"
AI_ACT_ENTRY_INTO_FORCE = date(2024, 8, 1)

# Supported languages (24 EU official). Default below is a Belgium-relevant
# subset; add more freely — the parser handles every language the Formex
# bundle ships.
ALL_EU_LANGUAGES = (
    "en", "nl", "fr", "de", "es", "it", "pt", "pl",
    "ro", "el", "sv", "fi", "da", "cs", "sk", "hu",
    "et", "lv", "lt", "sl", "bg", "hr", "mt", "ga",
)


@dataclass
class BuildConfig:
    languages: list[str] = field(default_factory=lambda: ["en", "nl", "fr"])
    include_article_full: bool = True
    include_recitals: bool = True
    include_annexes: bool = True
    # Output paths (resolved relative to CWD or to --output)
    output_dir: str = "dist"
    cache_dir: str = ".dataset_cache/raw"

    # Repository to publish to (override per env / CLI).
    hf_repo: str = "jeroenherczeg/eu-ai-act-2024-1689"
