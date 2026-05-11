"""Article/Recital/Annex-aware chunker.

Consumes a ParsedAct and emits a list of Chunk records ready to enrich + embed
+ upsert. Stays pure (no I/O, no DB) so it can be unit-tested deterministically.

Chunk types emitted:

  paragraph         — one article paragraph (with its sub-points inlined)
  article_full      — the whole article rendered as one chunk; used as parent
                      so a high-scoring paragraph can hydrate its full article
                      at answer time
  recital           — one recital
  annex_item        — one annex section / area (e.g. each of Annex III's 8 areas)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from eu_ai_act_dataset.models import Annex, Article, ParsedAct, Recital


# --- public dataclass --------------------------------------------------------


@dataclass
class Chunk:
    document_id: str
    chunk_type: str
    citation_label: str
    structure_path: str                   # natural-key suffix used for upsert
    structure: dict[str, Any]             # serialized to JSONB
    text: str
    language: str
    source_url: str
    source_type: str = "regulation"
    parent_structure_path: str | None = None
    references_articles: list[int] = field(default_factory=list)
    interprets_articles: list[int] = field(default_factory=list)
    defined_terms: list[str] = field(default_factory=list)
    risk_category: list[str] = field(default_factory=list)
    actor_roles: list[str] = field(default_factory=list)
    obligation_type: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    effective_from: date | None = None    # populated by enrich
    transitional: bool = False
    token_count: int | None = None        # filled by embedder

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def embed_input(self) -> str:
        """Embedding input — we prefix the citation label to bias retrieval
        toward exact-citation queries ("Article 9(2)(a)")."""
        return f"{self.citation_label}\n\n{self.text}"


# --- chunking entry point ---------------------------------------------------


def chunk_act(act: ParsedAct, *, source_type: str = "regulation") -> list[Chunk]:
    out: list[Chunk] = []
    out.extend(_chunk_recitals(act))
    out.extend(_chunk_articles(act, source_type=source_type))
    out.extend(_chunk_annexes(act, source_type=source_type))
    return out


# --- recitals ---------------------------------------------------------------


def _chunk_recitals(act: ParsedAct) -> list[Chunk]:
    out: list[Chunk] = []
    for r in act.recitals:
        text = r.text
        interprets = _extract_article_refs(text)
        out.append(
            Chunk(
                document_id=act.document_id,
                chunk_type="recital",
                citation_label=f"Recital ({r.number}) AI Act",
                structure_path=f"rec:{r.number}",
                structure={"recital_no": r.number},
                text=text,
                language=act.language,
                source_url=act.source_url,
                source_type="regulation",
                references_articles=interprets,
                interprets_articles=interprets,
            )
        )
    return out


# --- articles ---------------------------------------------------------------


def _chunk_articles(act: ParsedAct, *, source_type: str) -> list[Chunk]:
    """Emit one `article_full` chunk per article and one `paragraph` chunk per
    article paragraph. Paragraph chunks link back to the article_full via
    parent_structure_path so retrieval can hydrate the full article when ≥2
    paragraphs of the same article hit."""
    out: list[Chunk] = []
    for article in act.articles:
        parent_path = f"art:{article.number}"
        article_struct = _article_structure(article)

        out.append(
            Chunk(
                document_id=act.document_id,
                chunk_type="article_full",
                citation_label=_article_label(article, paragraph_no=None, point=None),
                structure_path=parent_path,
                structure={**article_struct, "scope": "article_full"},
                text=article.raw_text,
                language=act.language,
                source_url=act.source_url,
                source_type=source_type,
                references_articles=_extract_article_refs(article.raw_text, exclude={article.number}),
            )
        )

        unnumbered_idx = 0
        for p in article.paragraphs:
            text = p.full_text()
            if not text:
                continue
            # Structure path uses Formex IDENTIFIER position (structural, stable
            # across languages) rather than NO.PARAG (the displayed number, which
            # disagrees with position when a translation has a clerical typo —
            # e.g. NL Article 73 paragraph 10 displays as "11."). citation_label
            # below still uses paragraph_no so users see the citation exactly as
            # it appears in the official text of their language.
            if p.position:
                struct_path = f"{parent_path}/par:{p.position}"
            elif p.paragraph_no:
                struct_path = f"{parent_path}/par:{p.paragraph_no}"
            else:
                unnumbered_idx += 1
                struct_path = f"{parent_path}/par:0.{unnumbered_idx}"
            out.append(
                Chunk(
                    document_id=act.document_id,
                    chunk_type="paragraph",
                    citation_label=_article_label(article, paragraph_no=p.paragraph_no or None, point=None),
                    structure_path=struct_path,
                    structure={
                        **article_struct,
                        "paragraph_no": p.paragraph_no or None,
                        "scope": "paragraph",
                    },
                    text=text,
                    language=act.language,
                    source_url=act.source_url,
                    source_type=source_type,
                    parent_structure_path=parent_path,
                    references_articles=_extract_article_refs(
                        text, exclude={article.number}
                    ),
                )
            )
    return out


def _article_structure(article: Article) -> dict[str, Any]:
    path = article.division_path
    return {
        "article_no": article.number,
        "title_no": path.title_no(),
        "chapter_no": path.chapter_no(),
        "section_no": path.section_no(),
    }


def _article_label(article: Article, paragraph_no: int | None, point: str | None) -> str:
    base = f"Art. {article.number}"
    if paragraph_no:
        base += f"({paragraph_no})"
    if point:
        base += f"({point})"
    return f"{base} AI Act"


# --- annexes ----------------------------------------------------------------


def _chunk_annexes(act: ParsedAct, *, source_type: str) -> list[Chunk]:
    out: list[Chunk] = []
    for annex in act.annexes:
        for idx, sec in enumerate(annex.sections, start=1):
            text = _render_annex_section(annex, sec)
            if not text:
                continue
            section_token = sec.number or f"s{idx}"
            sec_path = f"anx:{annex.number}/sec:{section_token}"
            label = f"Annex {annex.number}"
            if sec.number:
                label += f", point {sec.number}"
            elif sec.label and sec.label != "Section":
                label += f", {sec.label.lower()}"
            label += " AI Act"
            structure = {
                "annex_no": annex.number,
                "annex_section": sec.number or section_token,
            }
            refs = _extract_article_refs(text)
            risk = _risk_for_annex_iii(annex.number, sec.number)
            out.append(
                Chunk(
                    document_id=act.document_id,
                    chunk_type="annex_item",
                    citation_label=label,
                    structure_path=sec_path,
                    structure=structure,
                    text=text,
                    language=act.language,
                    source_url=act.source_url,
                    source_type=source_type,
                    references_articles=refs,
                    risk_category=risk,
                )
            )
    return out


def _render_annex_section(annex: Annex, sec) -> str:  # noqa: ANN001 — AnnexSection
    header_bits = [f"Annex {annex.number}"]
    if annex.title:
        header_bits.append(f"— {annex.title}")
    header = " ".join(header_bits)
    sub = sec.label
    if sec.title and sec.title != sec.label:
        sub = f"{sec.label}: {sec.title}"
    return f"{header}\n{sub}\n\n{sec.text}".strip()


def _risk_for_annex_iii(annex_no: str, sec_no: str | None) -> list[str]:
    """Every Annex III item is, by definition, a high-risk AI system area."""
    if annex_no == "III":
        return ["high_risk"]
    return []


# --- cross-reference extraction --------------------------------------------


# Match "Article 9", "Articles 9 and 10", "Article 9(2)(a)" — capture article numbers only.
_ARTICLE_REF_RE = re.compile(r"Article[s]?\s+(\d+)(?:\s*(?:to|–|—|-|and)\s*(\d+))?", re.IGNORECASE)


def _extract_article_refs(text: str, *, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    found: list[int] = []
    seen: set[int] = set()
    for m in _ARTICLE_REF_RE.finditer(text):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start or end - start > 30:  # guard against false ranges
            end = start
        for n in range(start, end + 1):
            if n in exclude or n in seen:
                continue
            seen.add(n)
            found.append(n)
    return found
