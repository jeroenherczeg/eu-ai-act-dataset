"""Fetch the EU AI Act Formex bundle from EUR-Lex Cellar.

Resolve CELEX → Cellar `branch` notice → FMX4 manifestations by language →
zipped Formex bundle. Each bundle contains the main act body + one XML file
per annex. We extract everything into a per-language cache directory.

Cellar caches generated artefacts but returns 202 "Accepted" with an empty
body while it warms the cache. We poll until 200, bounded by max_wait_s.

For CI runs, the cache directory is empty; for local runs, we re-use it.
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

from eu_ai_act_dataset.config import AI_ACT_CELLAR_ID, AI_ACT_CELEX

log = logging.getLogger(__name__)

CELLAR_NOTICE_URL = "https://publications.europa.eu/resource/cellar/{cellar_id}"
ISO_2_TO_3 = {
    "en": "ENG", "nl": "NLD", "fr": "FRA", "de": "DEU", "es": "SPA",
    "it": "ITA", "pt": "POR", "pl": "POL", "ro": "RON", "el": "ELL",
    "sv": "SWE", "fi": "FIN", "da": "DAN", "cs": "CES", "sk": "SLK",
    "hu": "HUN", "et": "EST", "lv": "LAV", "lt": "LIT", "sl": "SLV",
    "bg": "BUL", "hr": "HRV", "mt": "MLT", "ga": "GLE",
}
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
USER_AGENT = "eu-ai-act-dataset/0.1 (+https://github.com/; data-pipeline)"


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
    """Fetch one Formex bundle per requested language, caching to disk.

    The bundle's content_hash is taken over the *zip*, not the body — that way
    the hash also reflects changes to annex files.
    """
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
        notice_url = CELLAR_NOTICE_URL.format(cellar_id=AI_ACT_CELLAR_ID)
        log.info("fetching Cellar notice for %s lang=%s", AI_ACT_CELEX, iso3)
        notice = _http_get_polling(
            notice_url,
            accept="application/xml;notice=branch",
            accept_language=iso3.lower(),
        )
        doc_zip_url = _find_fmx4_doc_url(notice.content, iso3)
        if not doc_zip_url:
            raise RuntimeError(
                f"no FMX4 manifestation for {AI_ACT_CELEX} lang={iso3} in Cellar notice"
            )
        log.info("downloading Formex bundle: %s", doc_zip_url)
        zip_resp = _http_get(doc_zip_url)
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
    url: str, *, accept: str, accept_language: str | None, max_wait_s: int | None = None
) -> httpx.Response:
    """Cellar may return 202 with an empty body while warming its cache.
    Poll until 200, with bounded backoff."""
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
                f"Cellar returned {r.status_code} with empty body for {url} after {max_wait_s}s"
            )
        log.info("Cellar returned %s for %s; retrying in %.0fs", r.status_code, url, delay)
        time.sleep(delay)
        delay = min(delay * 1.5, 15.0)


def _find_fmx4_doc_url(notice_xml: bytes, iso3: str) -> str | None:
    root = etree.fromstring(notice_xml)
    for man in root.iter("MANIFESTATION"):
        if (man.get("manifestation-type") or "").lower() != "fmx4":
            continue
        token = f".{iso3}.fmx4".lower()
        if not any(token in (v or "").lower() for v in (el.text for el in man.iter("VALUE"))):
            continue
        for item in man.iter("ITEM"):
            for value in item.iter("VALUE"):
                if value.text and "/DOC_" in value.text:
                    return value.text.strip()
    return None


def _largest_act_xml(directory: Path) -> Path:
    """Pick the act body file (largest .fmx.xml that isn't doc/toc metadata)."""
    candidates = [
        p
        for p in directory.glob("*.fmx.xml")
        if not p.name.endswith((".doc.fmx.xml", ".toc.fmx.xml"))
    ]
    if not candidates:
        raise FileNotFoundError(f"no body .fmx.xml in {directory}")
    return max(candidates, key=lambda p: p.stat().st_size)
