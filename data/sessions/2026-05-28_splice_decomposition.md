# Splice cost decomposition — 2026-05-28 session

Raw transcript: `2026-05-28_splice_decomposition.jsonl` (full JSONL).

Companion in NOTES: §5p, §5q, §5r.

## Where we started

§5o (last session): proved the 4/60 residual after anchor v2 came from
*splicing distractor siblings*, not from gold splice. Gold-only mode hit
58/60 (raw-matching). Open question: what specifically about sibling
splice is costly? Three hypotheses:

1. **Drift direction is correlated across siblings** → shared off-axis
   signal misleads attention
2. **Drift magnitude is too large** at cos=0.8 vs fresh
3. Binary "spliced at all" property — anything that overwrites K_fresh
   100% costs accuracy

User pushed direction (1) — "give each chunk a unique anchor so drifts
decorrelate" — but agreed to verify by measuring sibling drift cos first.

## Step 1: measure sibling drift correlation under anchor v2

Wrote `scripts/14_drift_correlation.py`. Per-head cos between
`(K_fresh - K_shifted)` at sib0 and sib1 positions.

```
layer  ‖dK_gold‖  ‖dK_sib0‖  ‖dK_sib1‖  cos(sib0,sib1)
 L3     0.004      5.6         7.5          0.86
 L11    0.009      9.7        13.6          0.79
 L23    0.006      6.3         8.6          0.84
```

Sibling drifts share a strong direction (~0.80). Hypothesis 1 looked
strongly supported.

## Step 2: random-anchor cache to break the shared direction (§5p)

`build_random_anchor_chunk_cache` in `chunk_cache.py`. Per chunk_i,
sample M random vocab tokens seeded by `(case_id, chunk_id)`. Cache via
`[random_i + chunk_i]` short forward. Splice with `random_i` inserted
fresh in assembly before chunk_i.

Result: **51/60** — regressed by 4/60 on bookshop vs anchor v2 (55).

Re-measured drift correlation under random-anchor cache
(`scripts/15_drift_correlation_rnd.py`): `cos(sib0, sib1)` = 0.78
(basically unchanged from 0.80). The drift direction is robust to anchor
identity — driven by something geometric in deep-layer short-forward K,
not by the prefix content.

Also: gold drift jumped from ~0 to 1.7–3.2 because random anchor breaks
the prefix-match property that gave gold cos=1.0 under v2 (in v2, the
sink IS the global sink in assembly; under random, gold has extra
context before its anchor in assembly).

**Hypothesis 1 disconfirmed.**

## Step 3: anchor M sweep (§5q)

If direction is robust, maybe magnitude matters. Sweep
`--anchor_M ∈ {4, 8, 16, 32}` on sink_oracle_k3.

Accuracy: 54-55 / 54 / 54 / 54 — completely flat.

Drift magnitude at M=32: sib0 shrunk 15%, gold drift grew. Accuracy
didn't move either way.

**Hypothesis 2 disconfirmed** at the 15% magnitude scale.

## Step 4: α-blend cliff (§5r)

`K_spliced = α · K_cached + (1−α) · K_fresh`. Added `alpha` to
`assemble.patched_full_attn` and `--reuse_cache` flag to script 12 so
the cache only builds once for the sweep.

```
α     sink_oracle_k3   per-q
1.0   54/60            1.12s
0.75  58/60            1.15s
0.5   58/60            1.16s
0.25  58/60            1.17s
0.0   58/60            1.20s
```

**Complete step function.** Any α<1.0 → 58/60. Mechanism: at α=1.0,
attention's Q sees only `Q · K_cached`, where K_cached has cos=0.8 vs
fresh — 20% off-axis is enough to misroute attention. Adding even 25%
fresh K provides the assembly-context signal that Q needs; cached
contribution becomes tolerated noise.

**Hypothesis 3 confirmed** — the cost is binary at α=1.0.

## Step 5: the value decomposition (user's question)

User asked: *if we use the splice format but skip the K/V cache, does
the model still judge correctly?*

This is `raw_oracle_k3`: same `[sink + 3 chunks + Q]` layout, but no
`patched_full_attn` (everything fresh). It's been there since §5h.

```
                                    tokens   per-q    acc
baseline (full prompt)              8147     2.81s    57/60
raw_oracle_k3 (format, no cache)    ~770     1.40s    58/60
sink_oracle_k3 α=0.5                ~770     1.16s    58/60
sink_oracle_k3 α=1.0                ~770     1.12s    54/60
```

Decomposing:

- **Short assembly + sink** (format alone): 2.81s → 1.40s = 2× speedup,
  +1/60 accuracy. This is classic RAG.
- **Cache K/V at α<1.0**: 1.40s → 1.16s = 17% extra speedup, 0 accuracy.
- **Cache K/V at α=1.0**: 1.40s → 1.12s = 25% extra speedup, **−4/60**.

**Most of ReAttention's win is the format change**, not the cache K/V
replacement. The cache K/V is edge optimization (and a footgun at α=1.0).

## Concrete demo (case 0 / q0, bookshop in Lisbon)

```
                Input tokens   Time     Output
baseline        8147           4.90s    "Linden Street"
splice α=0.5     788           1.43s    "...the best place to find..." (cut at 24 tokens)
                10.3× shorter  3.4× faster
```

Both correct on the underlying accuracy sweep (58/60).

## Verdict for the cache K/V idea

Partially-negative on the 8K MK suite. The cache K/V replacement
adds at most 17% per-query speedup at the cost of either nothing
(α<1.0) or 4/60 accuracy (α=1.0). The real wins are the format
change (short assembly) and sink prepend.

Where cache K/V might still earn its keep (untested):

- Longer chunks (chunk_size=512, 1024)
- Longer contexts (64K, 128K)
- Cross-chunk reasoning, not single-needle
- Tight retrieval noise regimes

The long-chunk benchmark (chunk_size=512, α∈{1.0, 0.5, 0.0}) was run
afterward and is documented in NOTES §5s. Result: the cache K/V's
contribution gets worse at chunk_size=512 — α=1.0 costs 6/60 (vs 4/60
at 256), and the α<1.0 speed advantage over raw drops from 17% to 0.7%.
The "longer chunks → cache K/V earns its keep" hypothesis was the wrong
direction.

## Files touched

- `src/sprag/chunk_cache.py`: `build_random_anchor_chunk_cache`
- `src/sprag/assemble.py`: `alpha` parameter on `patched_full_attn`
- `scripts/12_sink_mk.py`: `--cache_kind anchor_random`,
  `rnd_anchor_oracle_k3` mode, `--alpha`, `--reuse_cache`
- `scripts/14_drift_correlation.py`: new (drift cos under anchor v2)
- `scripts/15_drift_correlation_rnd.py`: new (drift cos under random anchor)
- `NOTES.md`: §5p, §5q, §5r
