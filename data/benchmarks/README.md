# External evaluation benchmarks

Downloaded 2026-05-29 for cross-method comparison of sprag (ReAttention /
short-assembly + sink). Raw dumps are gitignored (redownloadable); this index
is tracked. Total on disk ≈ 957 MB.

| dir | source | size | shape |
|-----|--------|------|-------|
| `longbench_v1/` | `THUDM/LongBench` (HF dataset) | 350 MB | 34 `.jsonl` subsets under `data/` |
| `longbench_v2/` | `THUDM/LongBench-v2` (HF dataset) | 465 MB | single `data.json`, 503 records |
| `rgb/` | github `chen700564/RGB` | 44 MB | `data/{en,zh}{,_fact,_int,_refine}.json` |
| `niah_gkamradt/` | github `gkamradt/LLMTest_NeedleInAHaystack` | 11 MB | 49 PaulGraham essays = haystack corpus |

## How to re-download

```bash
source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download as s; \
  s('THUDM/LongBench', repo_type='dataset', local_dir='data/benchmarks/longbench_v1'); \
  s('THUDM/LongBench-v2', repo_type='dataset', local_dir='data/benchmarks/longbench_v2')"
python -c "import zipfile; zipfile.ZipFile('data/benchmarks/longbench_v1/data.zip').extractall('data/benchmarks/longbench_v1')"
git clone --depth 1 https://github.com/chen700564/RGB.git data/benchmarks/rgb
git clone --depth 1 https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git data/benchmarks/niah_gkamradt
```

## Record schemas

- **LongBench v1** (`longbench_v1/data/<subset>.jsonl`): `input` (question),
  `context` (the long doc), `answers` (list), `length`, `dataset`, `language`,
  `all_classes`, `_id`. The `_e` suffixed files are the length-balanced
  "LongBench-E" variants.
- **LongBench v2** (`longbench_v2/data.json`): multiple-choice. `question`,
  `choice_A..D`, `answer`, `context`, `domain`, `sub_domain`, `difficulty`,
  `length`. Domains: Single-/Multi-Document QA, Long In-context Learning,
  Long-dialogue, Long Structured Data, Code Repository Understanding.
- **RGB** (`rgb/data/en.json` etc.): one JSON object per line — `query`,
  `answer` (list of aliases), `positive` (gold passages), `negative`
  (distractor passages). The `_fact`/`_int`/`_refine` files target
  counterfactual robustness / information integration / negative rejection.
- **NIAH**: `niah_gkamradt/needlehaystack/PaulGrahamEssays/*.txt` is the
  classic haystack corpus; needles/tester live in `needlehaystack/`.

## Relevance to sprag

Most directly comparable to our short-assembly + oracle-retrieval setup
(retrieve-then-assemble over a long doc):

- **LongBench v1 RAG/multi-doc QA subsets**: `multifieldqa_en`, `hotpotqa`,
  `2wikimqa`, `musique`, `qasper`, `narrativeqa`. `passage_retrieval_en` and
  `passage_count` directly probe retrieval/positioning like our MK suite.
- **LongBench v2 Multi-/Single-Document QA**: multiple-choice scoring is
  cleaner to grade than v1's free-form F1/ROUGE.
- **RGB**: the closest thing to our "gold + distractor siblings" footgun —
  `positive`/`negative` passage split is exactly the sibling-splice regime.

## Not downloaded

- **RULER** (`NVIDIA/RULER`): ships no pre-generated data — it generates
  synthetic NIAH/MK/MV tasks on demand via its own pipeline (needs a tokenizer
  + heavy deps). We already maintain a RULER-equivalent MK suite in `data/mk/`
  (8K and 32K). Clone + run `scripts/data/prepare.sh` if a standardized RULER
  config is needed for a paper comparison.
- **Comparison-method implementations** (TurboRAG, RACC, KVPress): user is
  sourcing these separately.
