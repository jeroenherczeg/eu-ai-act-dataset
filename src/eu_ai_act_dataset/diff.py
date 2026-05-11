"""Diff a freshly built dist/ against the version currently on HF Hub.

Renders a short summary fit for `GITHUB_STEP_SUMMARY` so a glance at the
Actions run tells you whether anything material changed. Catches:

  - row-count drift
  - bundle_sha256 change (the source itself changed)
  - schema digest change (we changed the public schema — bump dataset version)

No diff at all means the run was a no-op publish (HF will deduplicate).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

log = logging.getLogger(__name__)


def diff_against_hf(dist: Path, repo: str) -> str:
    """Return a markdown summary of what changed vs the latest HF version."""
    local = json.loads((dist / "dataset_info.json").read_text())

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        remote_path = api.hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename="dataset_info.json",
        )
        remote = json.loads(Path(remote_path).read_text())
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        return f"### {repo}\n\nNo prior version on HF (or it's inaccessible: `{exc}`). This run will create or initialize the dataset.\n"

    lines = [f"### Diff against {repo}", ""]

    if local["version"] == remote.get("version"):
        lines.append(f"⚠️  Local and remote both at `{local['version']}` — push will be a re-publish.")
    else:
        lines.append(f"**Version**: `{remote.get('version', '?')}` → `{local['version']}`")

    local_rows = local["rows"]
    remote_rows = remote.get("rows", 0)
    delta = local_rows - remote_rows
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    lines.append(f"**Rows**: {remote_rows} {arrow} **{local_rows}** ({delta:+d})")

    if local.get("schema_digest") != remote.get("schema_digest"):
        lines.append("**Schema changed** — consider a major version bump in your card.")

    local_shas = {(b["language"], b["sha256"]) for b in local.get("bundles", [])}
    remote_shas = {(b["language"], b["sha256"]) for b in remote.get("bundles", [])}
    changed_langs = sorted({lang for (lang, _) in local_shas - remote_shas})
    if changed_langs:
        lines.append(f"**Source content changed** for languages: {', '.join(changed_langs)}")
    else:
        lines.append("Source bundles unchanged.")

    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--against-hf", required=True, help="repo id like 'user/eu-ai-act'")
    parser.add_argument("--emit-summary", default=None, help="path; usually $GITHUB_STEP_SUMMARY")
    args = parser.parse_args()

    summary = diff_against_hf(args.dist, args.against_hf)
    print(summary)
    if args.emit_summary:
        with open(args.emit_summary, "a") as fh:
            fh.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
