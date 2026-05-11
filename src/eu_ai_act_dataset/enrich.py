"""Metadata enrichment for parsed AI Act chunks.

M1 is intentionally regex / lookup only — no LLM calls. The two valuable things
we can compute deterministically now are:

  1. effective_from (per Article 113 of the AI Act): the date a provision
     actually applies. Drives the "what's active now / what kicks in 2026 / 2027"
     filtering that users will ask about.

  2. defined_terms: tag chunks that mention an Article 3 defined term verbatim,
     so a query for "general-purpose AI model" can boost chunks that contain
     the official term.

LLM-assisted tagging of risk_category / actor_roles / obligation_type is
deferred to M4 when the eval harness can measure whether it actually helps
retrieval. Pre-eval tagging is fashion, not engineering.
"""

from __future__ import annotations

from datetime import date

from eu_ai_act_dataset.chunk import Chunk

# Reference: Article 113 of Regulation (EU) 2024/1689.
AIA_IN_FORCE = date(2024, 8, 1)              # 20th day after OJ publication
APPLY_PROHIBITIONS = date(2025, 2, 2)        # Chapters I-II (Articles 1-5)
APPLY_GPAI_GOV = date(2025, 8, 2)            # GPAI, governance, penalties (most), notifying authorities
APPLY_DEFAULT = date(2026, 8, 2)             # default applicability
APPLY_ANNEX_I_PRODUCTS = date(2027, 8, 2)    # Art. 6(1) high-risk classification for Annex I products

# Article-number ranges that apply from 2 Aug 2025.
_AUG2025_ARTICLES = (
    set(range(28, 40))   # Chapter III Section 4 — notifying authorities / notified bodies
    | set(range(51, 57)) # Chapter V — GPAI
    | set(range(64, 71)) # Chapter VII — Governance
    | {78}               # Art. 78 — Confidentiality
    | {99, 100}          # Chapter XII — Penalties (most; Art. 101 follows default)
)


def effective_from_for(article_no: int | None, paragraph_no: int | None = None) -> date | None:
    """Return the applicability date of a given Article/paragraph per Art. 113."""
    if article_no is None:
        return APPLY_DEFAULT  # safe default; recitals/annexes will use this
    if article_no in {6} and paragraph_no == 1:
        return APPLY_ANNEX_I_PRODUCTS
    if 1 <= article_no <= 5:
        return APPLY_PROHIBITIONS
    if article_no in _AUG2025_ARTICLES:
        return APPLY_GPAI_GOV
    return APPLY_DEFAULT


# Curated subset of Article 3 defined terms, in the casing the Act uses.
# Order matters: longer phrases first to win greedy matching.
DEFINED_TERMS: tuple[str, ...] = (
    "general-purpose AI model with systemic risk",
    "general-purpose AI model",
    "general-purpose AI system",
    "real-time remote biometric identification system",
    "post-remote biometric identification system",
    "biometric identification",
    "biometric categorisation",
    "emotion recognition system",
    "high-risk AI system",
    "AI system",
    "AI model",
    "authorised representative",
    "notified body",
    "notifying authority",
    "market surveillance authority",
    "fundamental rights",
    "provider",
    "deployer",
    "importer",
    "distributor",
    "operator",
    "placing on the market",
    "putting into service",
    "making available on the market",
    "conformity assessment",
    "CE marking",
    "post-market monitoring system",
    "training data",
    "input data",
    "testing data",
    "validation data",
    "substantial modification",
    "intended purpose",
    "reasonably foreseeable misuse",
    "serious incident",
)


def find_defined_terms(text: str) -> list[str]:
    """Return defined terms that appear in `text`, longest match first.

    Case-sensitive match on the Act's official casing. Short common words like
    "provider" or "deployer" will inevitably match a lot — that's intentional:
    these are the AI Act's terms-of-art and we want chunks containing them to
    be findable when a user searches for the term.
    """
    found: list[str] = []
    seen: set[str] = set()
    for term in DEFINED_TERMS:
        if term in text and term not in seen:
            seen.add(term)
            found.append(term)
    return found


def enrich_chunks(chunks: list[Chunk]) -> None:
    """Mutate chunks in place with effective_from + defined_terms."""
    for c in chunks:
        article_no = c.structure.get("article_no")
        paragraph_no = c.structure.get("paragraph_no")
        c.effective_from = effective_from_for(article_no, paragraph_no)
        c.transitional = c.structure.get("article_no") == 111
        c.defined_terms = find_defined_terms(c.text)
