"""Formex 06.02.1 parser for EU regulations and annexes.

This parser is intentionally narrow: it targets the Formex flavour the
Publications Office emits today for documents like Regulation (EU) 2024/1689.
It produces a `ParsedAct` containing recitals, articles (with paragraph and
point structure) and annexes.

The XML tags we walk:

  <ACT>
    <TITLE><TI><P>...</P></TI></TITLE>
    <PREAMBLE>
      <GR.CONSID>
        <CONSID><NP><NO.P>(1)</NO.P><TXT>...</TXT></NP></CONSID>
      </GR.CONSID>
    </PREAMBLE>
    <ENACTING.TERMS>
      <DIVISION>                              -- Chapter
        <TITLE><TI><P>CHAPTER I</P></TI><STI><P>...</P></STI></TITLE>
        [<DIVISION>...</DIVISION>]            -- nested Sections
        <ARTICLE IDENTIFIER="001">
          <TI.ART>Article 1</TI.ART>
          <STI.ART><P>Subject matter</P></STI.ART>
          <PARAG IDENTIFIER="001.002">
            <NO.PARAG>2.</NO.PARAG>
            <ALINEA>... <LIST TYPE="alpha"><ITEM><NP><NO.P>(a)</NO.P><TXT>...</TXT></NP></ITEM></LIST></ALINEA>
          </PARAG>
        </ARTICLE>
      </DIVISION>
    </ENACTING.TERMS>

Annexes live in sibling files with root <ANNEX>.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lxml import etree

from eu_ai_act_dataset.models import (
    Annex,
    AnnexSection,
    Article,
    DivisionPath,
    Paragraph,
    ParagraphPoint,
    ParsedAct,
    Recital,
)

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
# Language-agnostic numeral extractor: find the first roman or arabic numeral
# token in the TITLE/TI of a DIVISION or ANNEX. Works across EN/NL/FR/DE/…
# because every EU language puts the numeral after the type word
# (CHAPTER I / CHAPITRE I / HOOFDSTUK I / KAPITEL I, ANNEX III / ANNEXE III /
# BIJLAGE III / ANHANG III, …). The numeral is the only stable token.
_NUMERAL_RE = re.compile(r"\b([IVXLCDM]+|\d+)\b")
_DIVISION_DEPTH_KIND = ("chapter", "section", "subsection")


def parse_act(body_xml_bytes: bytes, *, document_id: str, language: str, source_url: str) -> ParsedAct:
    root = etree.fromstring(body_xml_bytes)
    if root.tag != "ACT":
        raise ValueError(f"expected <ACT> root, got <{root.tag}>")

    title = _first_text(root.find("TITLE")) or ""
    act = ParsedAct(
        document_id=document_id,
        title=title,
        language=language,
        source_url=source_url,
    )
    act.recitals = _parse_recitals(root)
    act.articles = _parse_articles(root)
    return act


def parse_annex(annex_xml_bytes: bytes) -> Annex | None:
    """Parse one <ANNEX> root file. Returns None if the file is not an annex
    (some bundle files are corrigenda or other supplementary docs)."""
    root = etree.fromstring(annex_xml_bytes)
    if root.tag != "ANNEX":
        return None

    title_el = root.find("TITLE")
    annex_number = _annex_number(title_el)
    annex_title = _annex_subtitle(title_el)

    annex = Annex(number=annex_number or "?", title=annex_title)
    contents = root.find("CONTENTS")
    if contents is None:
        return annex

    sections = list(contents.findall("GR.SEQ"))
    if sections:
        for sec in sections:
            annex.sections.append(_parse_annex_section(sec))
    else:
        # Annex III pattern: a top-level <LIST TYPE="ARAB"> whose ITEMs are the
        # numbered areas (1..N). Emit one section per area so risk-classification
        # questions can target them precisely.
        top_lists = contents.findall(".//LIST")
        emitted = False
        for lst in top_lists:
            type_attr = (lst.get("TYPE") or "").upper()
            if type_attr not in {"ARAB", "1"}:
                continue
            for item in lst.findall("./ITEM"):
                annex.sections.append(_annex_section_from_item(item))
                emitted = True
            if emitted:
                break
        if not emitted:
            # Last-ditch: take the whole CONTENTS as one section.
            annex.sections.append(
                AnnexSection(label=annex.title or "Annex", text=_normalize_text(contents))
            )

    annex.raw_text = "\n\n".join(
        (f"{s.label}{': ' + s.title if s.title else ''}\n{s.text}".strip() for s in annex.sections)
    )
    return annex


def parse_bundle(directory: Path, *, document_id: str, language: str, source_url: str) -> ParsedAct:
    """Load every .fmx.xml file in `directory`, dispatching by root tag.

    `*.doc.fmx.xml` and `*.toc.fmx.xml` are metadata and skipped.
    """
    body_file = _find_act_file(directory)
    act = parse_act(
        body_file.read_bytes(),
        document_id=document_id,
        language=language,
        source_url=source_url,
    )

    for path in sorted(directory.glob("*.fmx.xml")):
        if path == body_file or path.name.endswith((".doc.fmx.xml", ".toc.fmx.xml")):
            continue
        try:
            annex = parse_annex(path.read_bytes())
        except etree.XMLSyntaxError as exc:
            log.warning("failed to parse %s as annex: %s", path.name, exc)
            continue
        if annex is None:
            log.debug("file %s has no <ANNEX> root, skipping", path.name)
            continue
        act.annexes.append(annex)

    act.annexes.sort(key=_annex_sort_key)
    return act


# --- recitals ---------------------------------------------------------------


def _parse_recitals(root: etree._Element) -> list[Recital]:
    out: list[Recital] = []
    gr = root.find("PREAMBLE/GR.CONSID")
    if gr is None:
        return out
    for idx, consid in enumerate(gr.iter("CONSID"), start=1):
        number = _parse_recital_number(consid) or idx
        # NO.P is the "(N)" marker — extract the body from <TXT> when present,
        # else fall back to the whole CONSID minus the NO.P.
        txt = consid.find(".//TXT")
        if txt is not None:
            text = _normalize_text(txt)
        else:
            text = _text_excluding(consid, exclude_tags={"NO.P"})
        if text:
            out.append(Recital(number=number, text=text))
    return out


def _parse_recital_number(consid: etree._Element) -> int | None:
    no_p = consid.find(".//NO.P")
    if no_p is None or not (no_p.text or "").strip():
        return None
    m = re.search(r"\d+", no_p.text or "")
    return int(m.group(0)) if m else None


# --- articles ---------------------------------------------------------------


def _parse_articles(root: etree._Element) -> list[Article]:
    enacting = root.find("ENACTING.TERMS")
    if enacting is None:
        return []
    out: list[Article] = []
    _walk_divisions(enacting, DivisionPath(), out, depth=0)
    return out


def _walk_divisions(
    container: etree._Element,
    path: DivisionPath,
    out: list[Article],
    *,
    depth: int,
) -> None:
    """Recursively walk DIVISIONs accumulating the path stack; emit articles found.

    The DIVISION's "kind" (chapter / section / subsection) is inferred from
    nesting depth rather than the language-specific word in the TI text.
    This is the AI Act's actual layout in every translated language version.
    """
    for child in container:
        if not isinstance(child.tag, str):
            continue
        if child.tag == "DIVISION":
            new_path = DivisionPath(parts=list(path.parts) + [_division_label(child, depth)])
            _walk_divisions(child, new_path, out, depth=depth + 1)
        elif child.tag == "ARTICLE":
            out.append(_parse_article(child, path))


def _division_label(division: etree._Element, depth: int) -> tuple[str, str | None, str | None]:
    """Extract (kind, number, label) from a DIVISION's TITLE > TI/STI.

    The numeral is extracted with a language-agnostic regex; the kind is
    derived from nesting depth (0 → chapter, 1 → section, ≥2 → subsection).
    """
    ti_text = _first_text(division.find("TITLE/TI")) or ""
    sti_text = _first_text(division.find("TITLE/STI"))
    m = _NUMERAL_RE.search(ti_text)
    num = m.group(1) if m else None
    kind = _DIVISION_DEPTH_KIND[min(depth, len(_DIVISION_DEPTH_KIND) - 1)]
    return (kind, num, sti_text)


def _parse_article(article: etree._Element, path: DivisionPath) -> Article:
    raw_id = article.get("IDENTIFIER") or ""
    article_no = _strip_int(raw_id.split(".")[0]) or _parse_article_no_from_ti(article)
    title = _first_text(article.find("STI.ART"))
    art = Article(
        number=article_no or 0,
        title=title,
        division_path=path,
    )

    for parag in article.findall("PARAG"):
        art.paragraphs.append(_parse_paragraph(parag, article_no or 0))

    # Some short articles have an ALINEA directly under ARTICLE (no PARAG).
    if not art.paragraphs:
        for alinea in article.findall("ALINEA"):
            art.paragraphs.append(_parse_unnumbered_paragraph(alinea, article_no or 0))

    art.raw_text = _render_article(art)
    return art


def _parse_paragraph(parag: etree._Element, article_no: int) -> Paragraph:
    no_text = (parag.findtext("NO.PARAG") or "").strip().rstrip(".")
    para_no = int(no_text) if no_text.isdigit() else 0
    identifier = parag.get("IDENTIFIER")
    position = _position_from_identifier(identifier)
    text, points = _extract_paragraph_text_and_points(parag)
    return Paragraph(
        article_no=article_no,
        paragraph_no=para_no,
        text=text,
        points=points,
        identifier=identifier,
        position=position,
    )


def _position_from_identifier(identifier: str | None) -> int:
    """Extract the position-within-article from Formex IDENTIFIER "NNN.MMM".

    IDENTIFIER is structural and guaranteed unique within an ARTICLE; it's
    what we key on for structure_path so per-language display typos don't
    collide rows (see NL Article 73 paragraph 10 being mislabelled "11.").
    """
    if not identifier:
        return 0
    parts = identifier.split(".")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


def _parse_unnumbered_paragraph(alinea: etree._Element, article_no: int) -> Paragraph:
    text, points = _extract_paragraph_text_and_points(alinea, alinea_is_root=True)
    return Paragraph(
        article_no=article_no,
        paragraph_no=0,
        text=text,
        points=points,
    )


def _extract_paragraph_text_and_points(
    parag: etree._Element, *, alinea_is_root: bool = False
) -> tuple[str, list[ParagraphPoint]]:
    """Return (prose_text, list_of_points).

    Prose: everything inside <ALINEA> excluding <LIST>/<ITEM>.
    Points: each <ITEM><NP><NO.P>label</NO.P><TXT>text</TXT></NP></ITEM>.
    """
    alineas = [parag] if alinea_is_root else list(parag.findall("ALINEA"))
    opening_parts: list[str] = []   # alineas that come before the first LIST
    trailing_parts: list[str] = []  # alineas after the last LIST
    seen_list = False
    points: list[ParagraphPoint] = []

    for alinea in alineas:
        lists_in_alinea = alinea.findall(".//LIST")
        text_no_lists = _text_excluding(alinea, exclude_tags={"LIST"})
        if not seen_list and not lists_in_alinea:
            opening_parts.append(text_no_lists)
        elif lists_in_alinea:
            seen_list = True
            if text_no_lists:
                opening_parts.append(text_no_lists)
            for list_el in lists_in_alinea:
                for item in list_el.findall("./ITEM"):
                    np = item.find("NP")
                    if np is None:
                        continue
                    label = (np.findtext("NO.P") or "").strip()
                    text = _normalize_text(np.find("TXT")) or _normalize_text(np)
                    if text:
                        points.append(ParagraphPoint(label=label, text=text))
        else:
            trailing_parts.append(text_no_lists)

    prose = _normalize_whitespace("\n".join(s for s in opening_parts if s))
    # Trailing prose is preserved through full_text(): append after the points.
    if trailing_parts:
        trailing = _normalize_whitespace("\n".join(s for s in trailing_parts if s))
        if trailing:
            # Encode as a synthetic point with empty label so the chunker still emits it.
            points.append(ParagraphPoint(label="", text=trailing))
    return prose, points


def _render_article(article: Article) -> str:
    """Render the full article as plain text for the article_full chunk."""
    header = f"Article {article.number}"
    if article.title:
        header += f" — {article.title}"
    lines = [header]
    for p in article.paragraphs:
        if p.paragraph_no:
            lines.append(f"{p.paragraph_no}. {p.full_text()}")
        else:
            lines.append(p.full_text())
    return "\n\n".join(lines).strip()


# --- annexes ----------------------------------------------------------------


def _annex_number(title_el: etree._Element | None) -> str | None:
    """Extract the annex's roman numeral (or arabic number) from its TI text.

    Language-agnostic: works for ANNEX III (EN), ANNEXE III (FR), BIJLAGE III
    (NL), ANHANG III (DE), ALLEGATO III (IT), … — we don't care what the type
    word is, only the numeral that follows it.
    """
    if title_el is None:
        return None
    ti_text = _first_text(title_el.find("TI")) or ""
    m = _NUMERAL_RE.search(ti_text)
    return m.group(1).upper() if m else None


def _annex_subtitle(title_el: etree._Element | None) -> str | None:
    if title_el is None:
        return None
    return _first_text(title_el.find("STI"))


def _annex_section_from_item(item: etree._Element) -> AnnexSection:
    """Build an AnnexSection from one top-level <ITEM> in an Annex's numbered list.

    Used for Annex III (the 8 high-risk areas) and any annex laid out as a
    flat ARAB list rather than GR.SEQ groups.
    """
    np = item.find("NP")
    number = (np.findtext("NO.P") if np is not None else "").strip().rstrip(".") or None
    # Title = the first TXT under NP (the area's heading sentence).
    title = _normalize_text(np.find("TXT")) if np is not None else None
    # Body = the full ITEM rendered, including any nested LIST sub-points.
    text = _normalize_text(item)
    label = f"Point {number}" if number else "Item"
    return AnnexSection(label=label, number=number, title=title, text=text)


def _parse_annex_section(sec: etree._Element) -> AnnexSection:
    title_el = sec.find("TITLE")
    label_text = _first_text(title_el.find("TI")) if title_el is not None else None
    sub_title = _first_text(title_el.find("STI")) if title_el is not None else None

    # The body of an annex section is whatever lives outside its TITLE.
    body_parts: list[str] = []
    for child in sec:
        if not isinstance(child.tag, str) or child.tag == "TITLE":
            continue
        body_parts.append(_normalize_text(child))

    label = label_text or "Section"
    number = _section_number(label_text)
    return AnnexSection(
        label=label.strip(),
        number=number,
        title=sub_title,
        text=_normalize_whitespace("\n\n".join(p for p in body_parts if p)),
    )


def _section_number(label: str | None) -> str | None:
    if not label:
        return None
    m = re.match(r"^([IVXLCDM]+|[0-9]+|[A-Z])\.?\b", label.strip())
    return m.group(1) if m else None


def _annex_sort_key(a: Annex) -> int:
    """Roman annex numbers sort by their decimal value; fallbacks land last."""
    from eu_ai_act_dataset.models import _roman_to_int  # local to avoid public re-export

    v = _roman_to_int(a.number)
    return v if v is not None else 9999


# --- low-level helpers -------------------------------------------------------


def _normalize_text(el: etree._Element | None) -> str:
    if el is None:
        return ""
    # itertext() flattens descendants. We skip <NOTE> (footnotes) and processing
    # instructions because they describe sourcing, not normative content.
    parts: list[str] = []
    _collect_text(el, parts, skip={"NOTE"})
    return _normalize_whitespace("".join(parts))


def _text_excluding(el: etree._Element, exclude_tags: set[str]) -> str:
    parts: list[str] = []
    _collect_text(el, parts, skip={"NOTE", *exclude_tags})
    return _normalize_whitespace("".join(parts))


_LABEL_TAGS = {"NO.P", "NO.PARAG", "NO.SEQ"}  # numbered labels that need a separator from their sibling TXT


def _collect_text(el: etree._Element, out: list[str], skip: set[str]) -> None:
    if el.tag in skip:
        return
    if el.tag == "QUOT.START":
        out.append("“")
        return
    if el.tag == "QUOT.END":
        out.append("”")
        return
    if el.text:
        out.append(el.text)
    for child in el:
        if not isinstance(child.tag, str):
            if child.tail:
                out.append(child.tail)
            continue
        _collect_text(child, out, skip)
        if child.tag in _LABEL_TAGS:
            out.append(" ")  # ensure "4." doesn't bleed into the next sibling's text
        if child.tail:
            out.append(child.tail)


def _normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _first_text(el: etree._Element | None) -> str | None:
    if el is None:
        return None
    txt = _normalize_text(el)
    return txt or None


def _strip_int(s: str) -> int | None:
    s = (s or "").lstrip("0")
    return int(s) if s.isdigit() else None


def _parse_article_no_from_ti(article: etree._Element) -> int | None:
    ti = (article.findtext("TI.ART") or "").strip()
    m = re.search(r"Article\s+(\d+)", ti, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_act_file(directory: Path) -> Path:
    """Pick the body file from a Formex bundle directory (the largest .fmx.xml
    that isn't doc/toc metadata)."""
    candidates = [
        p
        for p in directory.glob("*.fmx.xml")
        if not p.name.endswith((".doc.fmx.xml", ".toc.fmx.xml"))
    ]
    if not candidates:
        raise FileNotFoundError(f"no body XML in {directory}")
    return max(candidates, key=lambda p: p.stat().st_size)
