"""Publish the exported dist/ bundle to a Hugging Face dataset repository.

Idempotent: creates the repo if it doesn't exist, uploads everything in dist/,
and creates an immutable tag for the snapshot version so older versions can
be loaded by ref. Safe to re-run on the same snapshot — HF will deduplicate
identical files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

log = logging.getLogger(__name__)


def publish(
    output_dir: Path,
    *,
    repo: str,
    snapshot_version: str,
    private: bool = False,
    commit_message: str | None = None,
    create_tag: bool = True,
) -> str:
    """Push `output_dir` to `repo` as the main branch state, then tag it.

    Returns the URL of the dataset repo.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a 'write' token at "
            "https://huggingface.co/settings/tokens and export it."
        )

    api = HfApi(token=token)

    log.info("ensuring HF dataset repo exists: %s", repo)
    create_repo(
        repo_id=repo,
        repo_type="dataset",
        token=token,
        private=private,
        exist_ok=True,
    )

    msg = commit_message or f"snapshot {snapshot_version}"
    log.info("uploading %s → %s (%s)", output_dir, repo, msg)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo,
        repo_type="dataset",
        commit_message=msg,
        # Don't include hidden files or local caches if any sneak in.
        ignore_patterns=[".*", "__pycache__", "*.pyc"],
    )

    if create_tag:
        try:
            api.create_tag(
                repo_id=repo,
                repo_type="dataset",
                tag=snapshot_version,
                tag_message=f"Snapshot {snapshot_version}",
            )
            log.info("created tag %s on %s", snapshot_version, repo)
        except Exception as exc:  # tag already exists, etc. — not fatal
            log.warning("could not create tag %s: %s", snapshot_version, exc)

    return f"https://huggingface.co/datasets/{repo}"
