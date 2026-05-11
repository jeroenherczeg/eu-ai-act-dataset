"""Fetch the EU AI Act Formex bundle from EUR-Lex Cellar.

For a known regulation (and the AI Act is known) we can skip the branch-notice
round trip — that endpoint is the most aggressively WAF-challenged at Cellar
and returns 202 challenges for tens of seconds on cold IPs. Instead:

  1. Construct the per-language manifestation URI deterministically:
       http://publications.europa.eu/resource/oj/{OJ_ID}.{ISO3}.fmx4
  2. Fetch its RDF descriptor (small, cache-friendly, ~3 KB).
  3. Extract the `cdm:manifestation_has_item` link → a DOC_* URL.
  4. GET the DOC_* URL → the zipped Formex bundle.
  5. Extract; the largest .fmx.xml is the act body, the rest are annexes.

Each step is retried on transient errors. 202s from any step are polled until
they resolve to 200 (Cellar's WAF eventually clears).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from lxml import etree
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from eu_ai_act_dataset.config import AI_ACT_CELEX, AI_ACT_OJ_ID

log = logging.getLogger(__name__)

ISO_2_TO_3 = {
    "en": "ENG", "nl": "NLD", "fr": "FRA", "de": "DEU", "es": "SPA",
    "it": "ITA", "pt": "POR", "pl": "POL", "ro": "RON", "el": "ELL",
    "sv": "SWE", "fi": "FIN", "da": "DAN", "cs": "CES", "sk": "SLK",
    "hu": "HUN", "et": "EST", "lv": "LAV", "lt": "LIT", "sl": "SLV",
    "bg": "BUL", "hr": "HRV", "mt": "MLT", "ga": "GLE",
}
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
USER_AGENT = "eu-ai-act-dataset/0.1 (+https://github.com/jeroenherczeg/eu-ai-act-dataset)"

# RDF / CDM namespaces in the manifestation descriptor.
_CDM_NS = "http://publications.europa.eu/ontology/cdm#"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _manifestation_uri(iso3: str) -> str:
    return f"http://publications.europa.eu/resource/oj/{AI_ACT_OJ_ID}.{iso3}.fmx4"


@dataclass
class FetchedBundle:
    """One fetched, extracted Formex bundle for a given language."""

    language: str                       # ISO-639-1 ("en", "nl", "fr", ...)
    iso3: str                           # ISO-639-3 used by Cellar URLs
    directory: Path                     # cache dir containing the extracted xml files
    body_xml: bytes                     # bytes of the act body file
    body_path: Path
    bundle_zip_path: Path
    bundle_sha256: str                  # sha of the zip — drives versioning


def fetch_bundles(languages: list[str], cache_root: Path) -> list[FetchedBundle]:
    bundles: list[FetchedBundle] = []
    cache_root.mkdir(parents=True, exist_ok=True)
    for lang in languages:
        iso3 = ISO_2_TO_3.get(lang.lower())
        if not iso3:
            log.warning("unknown language code %r, skipping", lang)
            continue
        bundles.append(_fetch_one(lang, iso3, cache_root))
    return bundles


def _fetch_one(lang: str, iso3: str, cache_root: Path) -> FetchedBundle:
    dest = cache_root / lang
    dest.mkdir(parents=True, exist_ok=True)
    bundle_path = dest / "bundle.zip"

    if bundle_path.exists() and bundle_path.stat().st_size > 1024:
        log.info("reusing cached bundle: %s", bundle_path)
        zip_bytes = bundle_path.read_bytes()
    else:
        manifestation = _manifestation_uri(iso3)
        log.info("fetching manifestation RDF: %s", manifestation)
        rdf = _http_get_polling(manifestation, accept="application/rdf+xml")
        doc_zip_url = _extract_doc_url(rdf.content)
        if not doc_zip_url:
            raise RuntimeError(
                f"no DOC_* link in manifestation RDF for {AI_ACT_CELEX} lang={iso3}"
            )
        log.info("downloading Formex bundle: %s", doc_zip_url)
        zip_resp = _http_get_polling(doc_zip_url, accept="application/zip")
        zip_bytes = zip_resp.content
        bundle_path.write_bytes(zip_bytes)

    # Extract (idempotent).
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)

    body_path = _largest_act_xml(dest)
    return FetchedBundle(
        language=lang,
        iso3=iso3,
        directory=dest,
        body_xml=body_path.read_bytes(),
        body_path=body_path,
        bundle_zip_path=bundle_path,
        bundle_sha256=hashlib.sha256(zip_bytes).hexdigest(),
    )


# --- HTTP helpers ----------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
def _http_get(url: str, accept: str | None = None, accept_language: str | None = None) -> httpx.Response:
    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    if accept_language:
        headers["Accept-Language"] = accept_language
    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r


def _http_get_polling(
    url: str,
    *,
    accept: str,
    accept_language: str | None = None,
    max_wait_s: int | None = None,
) -> httpx.Response:
    """Cellar's WAF returns 202 challenges with empty bodies before serving the
    actual resource. Poll until 200, with bounded backoff."""
    if max_wait_s is None:
        max_wait_s = int(os.environ.get("CELLAR_MAX_WAIT_S", "180"))
    deadline = time.monotonic() + max_wait_s
    delay = 3.0
    while True:
        r = _http_get(url, accept=accept, accept_language=accept_language)
        if r.status_code == 200 and r.content:
            return r
        if r.status_code not in (202, 204) or time.monotonic() > deadline:
            r.raise_for_status()
            raise httpx.HTTPError(
                f"Cellar returned {r.status_code} for {url} after {max_wait_s}s"
            )
        log.info("Cellar returned %s (likely WAF challenge); retrying in %.0fs", r.status_code, delay)
        time.sleep(delay)
        delay = min(delay * 1.5, 15.0)


def _extract_doc_url(rdf_xml: bytes) -> str | None:
    """Pull the `cdm:manifestation_has_item` resource URL from a manifestation
    RDF descriptor. The link points at `cellar/{id}.NNNN.NN/DOC_N` — the
    physical artefact (a zip containing the Formex XML files)."""
    root = etree.fromstring(rdf_xml)
    rdf_resource_attr = f"{{{_RDF_NS}}}resource"
    for el in root.iter(f"{{{_CDM_NS}}}manifestation_has_item"):
        href = el.get(rdf_resource_attr)
        if href and "/DOC_" in href:
            return href.strip()
    # Fallback: the older `cdm:has` predicate also points to the same item.
    for el in root.iter(f"{{{_CDM_NS}}}has"):
        href = el.get(rdf_resource_attr)
        if href and "/DOC_" in href:
            return href.strip()
    return None


def _largest_act_xml(directory: Path) -> Path:
    """The act body is the largest .fmx.xml file in the bundle (annex files are
    smaller siblings; the .doc/.toc metadata files are excluded)."""
    candidates = [
        p
        for p in directory.glob("*.fmx.xml")
        if not p.name.endswith((".doc.fmx.xml", ".toc.fmx.xml"))
    ]
    if not candidates:
        raise FileNotFoundError(f"no body .fmx.xml in {directory}")
    return max(candidates, key=lambda p: p.stat().st_size)
