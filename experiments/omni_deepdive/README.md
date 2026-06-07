# Omni deep-dives — three mechanistic stress-tests (methodological honesty)

Three follow-up experiments that stress-test the *sub-mechanisms* behind the video /
cross-modal KV-reuse results in [`../../docs/OMNI_RESULTS.md`](../../docs/OMNI_RESULTS.md).
The headline finding of that doc — **reusing a KV cache built over the full context
beats recomputing over a retrieved subset** (the "global-memory bonus", `ours > fresh`)
— survives all three. Two *secondary* claims do **not**, and are corrected here.

| # | Sub-claim under test | Verdict |
|---|---|---|
| 2 | Cross-modal audio trace lives in **deep** layers | ✅ recall real, ❌ **depth localization wrong** — trace is read out in *early–mid* layers |
| 3 | Position-preserving reuse **≫** compaction ("repositioning hurts") | ❌ **NULL even under maximal M-RoPE shear** — claim dropped |
| 4 | Sink-dup harm = attention dilution (two sinks grab the mass) | ❌ **falsified** — it is a decode-trajectory failure (see `../coverage_sinkdup`) |

\#4 is a text-model (Qwen3.5-27B) diagnostic and lives with the text work in
[`../coverage_sinkdup`](../coverage_sinkdup); summarized above for completeness.

---

## #2 — Where does the cross-modal trace live? (layer-wise, `scripts/48_omni_layerwise.py`)

**Setup.** Video-MME, prebake video **+ audio**, then **drop the audio** at use
(audio-token KV physically gathered out — same protocol as the `OMNI_RESULTS` cross-modal
section). For each record we sweep a **cumulative-depth** cache: layers `[0, d)` take the
audio-prebaked (cached) KV, layers `[d, L)` take the audio-free (fresh) recompute. `d=0`
is exactly `fresh`, `d=L=48` is exactly `cached` (built-in identity gate). Gold-option NLL.
`n = 252`, cov100.

**Core result (confirms cross-modal recall):**

> mean Δ(cached − fresh) = **−0.100 NLL, SEM 0.017** (t ≈ 6, p < 1e-6); **70.2 %** of
> records are cached-better.

**Depth profile — the hypothesis revision.** We expected the audio trace to be woven into
*deep* layers (shallow ≈ 0, mid-deep peak). It is **not**: most of the recoverable
advantage is already realized by swapping the **early** layers to cached.

| layers swapped to cached `[0,d)` | fraction of full gap realized |
|---:|---:|
| first 4  | 36 % |
| first 8  | **68 %** |
| first 12 | 72 % |
| first 24 | **100 %** (plateaus) |

So the contextualized cross-modal imprint is **readable from early–mid layer KV (~L8–24)**,
not concentrated deep. Earlier single-record glimpses that looked "deep-only" were not
representative. **Wording correction for any writeup:** say *"the trace is read out in
early–mid layers"*, not *"woven in deep layers"*.

Aggregate: `scripts/agg.py`. Data: `data/omni_lw.json.s*` (16 shards).

---

## #3 — Does position-preserving reuse really beat compaction? (`scripts/47_omni_baselines.py`)

`ours` keeps each retained video t-group at its **original** M-RoPE temporal index (gaps
preserved); `ours_compact` repacks the kept groups to **contiguous** temporal indices
(InfLLM-style compaction). `OMNI_RESULTS.md` originally reported "repositioning hurts" from
a small (+0.006…+0.009) borderline-significant gap on EgoSchema at the most aggressive
coverage. We built the regime **designed to maximize the M-RoPE temporal-axis shear** and
re-tested:

- **Video-MME long clips** (real temporal span), `--n_frames 64` ⇒ **t_grid = 32** (double
  the EgoSchema resolution → larger compaction shifts),
- **aggressive eviction** `--coverages 10 20 30` (drop up to 90 % of groups),
- `n = 236`, identity gate `cov100` exact (compact − ours = 0.0000, 0 % worse).

**Verdict: NULL.** Even here the compaction penalty is a noise-level nudge.

| subset | compaction penalty `mean(compact − ours)` @cov10 (SEM) | t | amplifies as predicted? |
|---|---|---|---|
| ALL (n=236) | +0.0061 (0.0030) | ~2.0 | — |
| **Temporal** questions (n=27) | +0.0056 (0.0060) | ~0.9 | ❌ smaller, n.s. |
| **Long** videos (n=65) | +0.0078 (0.0042) | ~1.9 | ❌ |
| **Short** videos (n=87) | +0.0104 (cov20) | ~1.6 | ❌ as large as long |

A genuine temporal grid-shear would **concentrate** the penalty on temporal-reasoning
questions and long videos. It does the opposite — short videos pay as much as long, and the
temporal subset is *smaller* and not significant — so the residual +0.006 is **noise, not
shear**. Qwen3-Omni's temporal index is a coarse integer; a shift of ~29 positions is a
negligible phase change. **Conclusion: drop the "position-preserving ≫ compaction / direct
attack on ReKV-InfLLM" claim.** Compaction is essentially free on this model (this
*corroborates* the Session-5 `+0.0025` decomposition and *corrects* finding #2 in
`OMNI_RESULTS.md`).

**What stands.** The robust, large, monotone-decaying effect in the *same* runs is the
memory bonus `ours > fresh` — and it is bigger than ever here:

| subset | `ours − fresh` @cov10 (NLL) | acc fresh→ours @cov10 |
|---|---|---|
| ALL | **−0.229** | 0.34 → **0.42** (+8 pt) |
| Short videos | **−0.325** | 0.33 → **0.51** (+18 pt) |

Aggregate (with task-type / duration split): `scripts/agg2.py`. Data:
`data/omni_mrope_vid.json.s*` (hard rerun, n=236) and `data/omni_mrope_ego.json.s*` (the
original EgoSchema run, n=101, also null pooled).

---

## Takeaway

Strip away the flashy sub-mechanisms — deep-layer localization (#2), M-RoPE grid-shear
(#3), attention dilution (#4) — and each falls. What is left standing, and in fact
reinforced (#3 gives the largest accuracy lift yet, +18 pt on short videos), is the single
unified claim:

> A KV cache built over the **full context** is a compressed associative memory of it;
> position-preserving reuse on a subset beats recompute because it remembers the dropped
> context — across text long-docs, video frames, and cross-modal (audio→video). The bonus
> is largest when recompute is most context-starved and vanishes at full coverage.

Compaction-vs-original positioning is **not** part of that story on this model, and
cross-modal recall is an **early–mid layer** phenomenon, not a deep-layer one.

## Files
- `../../scripts/48_omni_layerwise.py` — #2 cumulative-depth cross-modal sweep
- `../../scripts/47_omni_baselines.py` — #3 (and the ReKV/MuKV baselines); `ARMS=fresh,ours,ours_compact`, `--n_frames 64 --coverages 10 20 30 100`
- `scripts/agg.py` — #2 depth-profile aggregate
- `scripts/agg2.py` — #3 aggregate with Video-MME `task_type` (temporal) + `duration` split
- `data/omni_lw.json.s*` — #2 (n=252) · `data/omni_mrope_vid.json.s*` — #3 hard (n=236) · `data/omni_mrope_ego.json.s*` — #3 EgoSchema (n=101)
