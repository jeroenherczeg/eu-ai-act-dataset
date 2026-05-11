# eu-ai-act-dataset

A reproducible build pipeline that turns the official EU AI Act Formex XML
into a structured, multilingual, retrieval-ready dataset and publishes it
to the Hugging Face Hub.

- **Source**: Regulation (EU) 2024/1689 (CELEX [`32024R1689`](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)), Formex 06.02.1 from the Publications Office Cellar.
- **Output**: a single parquet file with structural metadata (article/recital/annex numbers, `effective_from` per Article 113, cross-references, parallel multilingual structure).
- **Where it lands**: a Hugging Face dataset repo, configurable per build.

> See [`DATA_LICENSE`](./DATA_LICENSE) for the data terms (CC BY 4.0, derived from official EU sources).
> See [`LICENSE`](./LICENSE) for the code terms (MIT).

## What makes this dataset different

Most public AI Act datasets are either the raw text dump or a noisy PDF scrape.
This pipeline produces:

1. **Article-aware structural metadata** on every row — `article_no`, `paragraph_no`,
   `recital_no`, `annex_no`, plus `references_articles` and `interprets_articles`
   (recitals → the articles they explain).
2. **`effective_from`** per Article 113 of the AI Act, so users can filter
   "what's active now / next year / 2027" without re-deriving the mapping.
3. **Language-invariant `structure_path`** — equivalent provisions across EN /
   NL / FR / DE share the same key, making the multilingual rows a parallel
   corpus by construction.
4. **Snapshot versioning + bundle sha256** on every row, so old versions of
   the dataset can be reproduced bit-for-bit.
5. **Optional gold retrieval split** (see [`gold/`](./gold/)) for benchmarking
   recall@k and citation-correctness against this corpus.

## Quickstart

```bash
# 1) install
uv sync

# 2) build (EN+NL+FR by default; ~10 minutes if Cellar cold-starts)
uv run eu-ai-act-dataset build --output dist

# 3) sanity-check (CI runs this on every PR)
uv run eu-ai-act-dataset validate dist

# 4) inspect locally
python -c "import pyarrow.parquet as pq; t = pq.read_table('dist/ai_act_chunks.parquet'); print(t.schema); print(t.num_rows, 'rows')"

# 5) (optional) publish — needs HF_TOKEN with 'write' permission
export HF_TOKEN=hf_...
uv run eu-ai-act-dataset publish dist --repo jeroenherczeg/eu-ai-act-2024-1689
```

## Continuous publishing

`.github/workflows/build-and-publish.yml` runs the build weekly (Monday 06:00
UTC), on manual dispatch, and on pushes to `main` that touch the build code.
It pushes to the Hugging Face Hub if `HF_TOKEN` is configured as a repo secret.

**One-time setup before the action will publish:**

1. Create a Hugging Face write token at <https://huggingface.co/settings/tokens>.
2. In your GitHub repo: *Settings → Secrets and variables → Actions*
   - Secret: `HF_TOKEN` = `hf_...`
   - Variable: `HF_REPO` = `jeroenherczeg/eu-ai-act-2024-1689` (optional;
     defaults to the placeholder otherwise)
3. (Optional) Create the dataset repo on HF first if you want it private at
   creation — the action will otherwise create a public one on first publish.

PRs that touch source code trigger `validate-pr.yml` which builds EN-only and
runs `validate` without publishing. The built `dist/` is uploaded as an
artifact for reviewers.

## Repo layout

```
src/eu_ai_act_dataset/
  fetch.py          # Cellar resolver + bundle download with 202-polling
  parsers/formex.py # Formex 06.02.1 → ParsedAct
  chunk.py          # ParsedAct → list[Chunk] (paragraph / article_full / recital / annex_item)
  enrich.py         # effective_from per Art. 113 + Article-3 defined terms
  export.py         # chunks → parquet + sources.csv + dataset_info.json + dataset card
  validate.py       # sanity invariants (row counts, schema digest, parallel structure)
  diff.py           # diff a built dist/ against the live HF version
  publish.py        # push dist/ to a HF dataset repo, tag the snapshot
  build.py          # the orchestrator
  cli.py            # `eu-ai-act-dataset` typer entrypoint
gold/
  retrieval_eval.yaml  # stub for the gold retrieval split
.github/workflows/
  build-and-publish.yml
  validate-pr.yml
DATASET_CARD.md.tmpl   # template; export renders this to dist/README.md
```

## Customization knobs

| Knob | Where |
|---|---|
| Languages | `--lang en --lang nl ...` on the build command (24 official EU languages supported) |
| Exclude annexes / recitals / article_full chunks | `--no-annexes` / `--no-recitals` / `--no-article-full` |
| Snapshot version tag | `--snapshot vYYYY-MM-DD` (auto-derived from build date if blank) |
| HF target repo | `--repo jeroenherczeg/eu-ai-act-2024-1689` (default) or `HF_REPO` env / GH variable |
| Cache directory | `--cache <path>` (re-runs are free once the bundle is cached) |

## Not legal advice

This dataset is informational. It must not be used to determine legal
compliance without human review. See [`DATA_LICENSE`](./DATA_LICENSE) for
the full disclaimer.
