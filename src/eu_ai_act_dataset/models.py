"""In-memory dataclasses produced by parsers and consumed by the chunker.

Distinct from the SQLAlchemy models and from the API Pydantic models — these
are just the parsing intermediate form. They carry enough provenance for the
chunker to compute citation_label, structure_path, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DivisionPath:
    """A hierarchical position in the act: list of (kind, number, label) entries.

    Example: [('chapter', 'I', 'GENERAL PROVISIONS'), ('section', '2', 'Definitions')]
    """

    parts: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    def title_no(self) -> int | None:
        return self._roman_or_int_for("title")

    def chapter_no(self) -> int | None:
        return self._roman_or_int_for("chapter")

    def section_no(self) -> int | None:
        return self._roman_or_int_for("section")

    def _roman_or_int_for(self, kind: str) -> int | None:
        for k, num, _ in self.parts:
            if k == kind and num is not None:
                return _roman_to_int(num)
        return None


@dataclass
class Recital:
    number: int                     # 1-180
    text: str                       # normalized plain text
    division_path: DivisionPath = field(default_factory=DivisionPath)


@dataclass
class ParagraphPoint:
    """A single (a) / (b) / (i) point inside an article paragraph."""

    label: str                      # e.g. "(a)", "(i)"
    text: str


@dataclass
class Paragraph:
    article_no: int                 # 1-113
    paragraph_no: int               # the displayed NO.PARAG, e.g. 2 from "2."
    text: str                       # joined ALINEA prose without the points
    points: list[ParagraphPoint] = field(default_factory=list)
    identifier: str | None = None   # raw "001.002"

    def full_text(self) -> str:
        """Display form: prose, then point lines, then any trailing subparagraph
        (carried as a point with empty label so order is preserved)."""
        if not self.points:
            return self.text
        lines = [self.text] if self.text else []
        for p in self.points:
            lines.append(f"{p.label} {p.text}".strip() if p.label else p.text)
        return "\n".join(lines).strip()


@dataclass
class Article:
    number: int                     # 1-113
    title: str | None = None        # STI.ART text (subject heading)
    paragraphs: list[Paragraph] = field(default_factory=list)
    division_path: DivisionPath = field(default_factory=DivisionPath)
    raw_text: str = ""              # full article text including title (for article_full chunk)


@dataclass
class AnnexSection:
    """A logical sub-unit inside an annex (a GR.SEQ or top-level numbered group)."""

    label: str                      # human label, e.g. "Section A" or "Point 4"
    number: str | None = None       # e.g. "4" (Annex III area number) or "A"
    title: str | None = None        # section heading
    text: str = ""                  # body text


@dataclass
class Annex:
    number: str                     # "I" .. "XIII" (roman as written)
    title: str | None = None        # STI text
    sections: list[AnnexSection] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class ParsedAct:
    document_id: str
    title: str
    language: str
    recitals: list[Recital] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)
    annexes: list[Annex] = field(default_factory=list)
    source_url: str = ""


_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int | None:
    s = (s or "").strip().upper()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if not all(c in _ROMAN_MAP for c in s):
        return None
    total = 0
    prev = 0
    for c in reversed(s):
        v = _ROMAN_MAP[c]
        total += -v if v < prev else v
        prev = v
    return total


PointStyle = Literal["alpha", "roman", "numeric"]
