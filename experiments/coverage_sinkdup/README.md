# Why "cached ≥ fresh" in the coverage runs — mechanism + a sink-duplication fix

This documents a correction to the A3B / 27B **cached-vs-fresh coverage** results
(`scripts/30_bigmodel_coverage.py`, `scripts/31_hybrid_coverage.py`). The original
runs reported "cached ≥ fresh at every coverage." Diagnosis shows that conclusion
needs **two corrections**: a mechanism re-interpretation, and a measurement-bug fix
that was inflating the high-coverage cells.

## Setup (what cached vs fresh actually are)

Per record: `target` = the chunk containing the gold answer, **always kept**.
Coverage `c` keeps the `c%` chunks immediately *before* the target. Layout =
`[sink] + kept ctx chunks + target`, compacted to contiguous positions.

- **fresh** = recompute KV over the kept subset only (each kept chunk attends to
  the kept subset before it).
- **cached** = take each kept chunk's KV from the *one full-document forward*
  (each token attended to the entire preceding document, incl. the dropped
  chunks), then `shift_rope` to the compact position.

## Correction 1 — cached > fresh is INFORMATION ASYMMETRY (= the video global-memory bonus)

cached's KV carry a **trace of the dropped chunks** (they were built over the full
doc); fresh's do not. So "cached > fresh" is not splice magic — it is the **same
global-memory bonus** seen in the video KV experiments (`video-kv-omni` branch):
reuse of a cache built over full context beats recompute over a starved subset.

Per dataset it tracks hop-count (A3B-Instruct diag, `scripts/34_a3b_diag.py`,
`data/a3b_diag.s*.json`, degen=0 in both arms):

- **hotpotqa (2-hop)**: cleanest positive (+0.04…+0.095).
- **musique (3–4 hop), low coverage**: cached is **negative** — the partial trace
  is *incomplete*, so cached confidently follows a stale reasoning chain to a wrong
  answer ("hallucinate from trace"), while fresh more often extracts the answer
  that is sitting in the kept target chunk. **Double-edged trace**: the dropped-
  context memory *helps* when sufficient (2-hop), *misleads* when partial (3-4 hop).
  This is NOT degeneration and NOT fresh-abstention (verified by reading gens).

## Correction 2 — a sink-duplication artifact inflated the high-coverage cells

When chunk0 (`a_start==0`) enters the kept set (i.e. high coverage, esp. c100), the
explicit sink `doc[0:M]` **duplicates** chunk0's head in the compact assembly. The
shared assembly feeds both arms, but **fresh recomputes full-attn over the malformed
double-sink**; on the thinking model Qwen3.5-27B this pushes decode into a verbose
`<think>…` that never closes within budget → empty after `strip_think` → scored
wrong. cached's full-attn KV come from the clean full-doc forward and are unaffected,
so cached looks like a big winner.

Diagnosis (`scripts/35_q27_diag.py`, `data/q27_diag.s*.json`, n=231 @ c100, with an
added `fresh_nodup` control arm + gen dump):

| arm | acc @ c100 |
|---|---|
| cached | 0.749 |
| fresh (duplicated sink) | 0.597 |
| fresh_nodup (de-duplicated) | 0.680 |

Δ(cached−fresh)=+0.152; **the sink-dup explains +0.082 (~54%)**; residual +0.069.
degen=0 in all arms (it's open-`<think>`, not collapse). 26 cells rescued (fresh
0→1 on de-dup) vs 7 broke. A3B-Instruct (non-thinking) is far less affected → its
c100 anomaly was only +0.022.

## The fix and the clean curves

Fix (`scripts/36_a3b_fix.py`, `scripts/37_q27_fix.py`): **do not add the explicit
sink when chunk0 is already kept** — chunk0's natural doc-start head serves as the
attention sink, no duplication. (Sanity: c100 assembly is exactly M tokens shorter.)

**Δacc(cached − fresh), original (buggy) → fixed:**

| cov | A3B orig | **A3B fixed (n=231)** | 27B orig | **27B fixed (n=234)** |
|----:|---------:|----------------------:|---------:|----------------------:|
|   0 |  +0.004  | **+0.004** |  +0.068  | **+0.068** |
|  25 |  +0.026  | **+0.026** |  +0.034  | **+0.034** |
|  50 |  +0.009  | **+0.009** |  −0.013  | **−0.013** |
|  75 |  +0.043  | **+0.017** |  +0.026  | **−0.021** |
| 100 |  +0.022  | **−0.022** |  +0.145  | **−0.026** |

(Data: `data/a3b_cov_fix.s*.json`, `data/q27_cov_fix.s*.json`.)

The fix moves **only** the cells where chunk0 enters ctx (c75/c100); **c0/c25/c50
are bit-for-bit unchanged** (27B shift = 0.000 at all three — the fixed numbers are
*literally equal* to the original). That surgical signature is itself the proof: the
high-coverage cache>fresh was the sink-dup artifact, the low-coverage cache>fresh is
the genuine memory bonus.

## Takeaway

After the fix, **text cache − fresh is a monotone-decaying memory bonus**: positive
at low–mid coverage (real dropped-context memory), crossing to ~0 / slightly
negative at full coverage (no dropped context to remember, only the splice/RoPE
drift cost predicted in `NOTES.md` §5j). This matches the video / cross-modal curve
shape. **Do not advertise "cached ≥ fresh at all coverages"** — the high-coverage
part was a measurement artifact. The defensible, unified claim:

> Reusing a KV cache **built over the full context** beats recomputing over a
> retrieved subset because the cache carries associative memory of the dropped
> context — across text long-docs, video frames, and cross-modal (audio→video).
> The bonus is largest when the recompute baseline is most context-starved and
> vanishes at full coverage. The trace is double-edged: it helps when sufficient
> (2-hop) and misleads when partial (3–4 hop).

## Scope boundary (read before generalizing)

This premium is bounded by the **contextual information gap**. It exists only when
the cache is built over a *unified* context and inference uses a *subset* of it:

- **(a) multi-query over unified long documents**, and **(b) streaming video with
  frame eviction** — the cache was built over the whole doc/clip, so it carries the
  dropped-context trace. ✅
- **Vanilla corpus-RAG (independently pre-baked chunks)**: cached and fresh face the
  *same* token set with *zero* information asymmetry (no "unselected context" was
  ever co-encoded), so the approach **converges to a latency-only speedup baseline,
  cached ≈ fresh** (cf. `NOTES.md` indep-cache / RGB). ❌ — do not claim an accuracy
  win here.

## Prior work (positioning)

- **TurboRAG** — isolated-chunk caching: pre-bake each chunk's KV independently,
  concatenate at inference (only attention-mask / position-ids touched). Goal is TTFT;
  quality-neutral by design. No dropped-context, hence no information-gap premium.
- **InfLLM / ReKV** — block-level KV retrieval with **compaction** (kept blocks are
  packed contiguously in memory), which scrambles the original absolute (M-)RoPE grid.
  Our `*_origpos` vs `*_compact` arms show **position-preserving reuse beats compaction**
  (repositioning hurts, esp. multi-hop / temporal) — a direct attack on their design.
- **MuKV** — dual-signal (attention + spectral) pruning; its selection win is
  **query-driven** (online, not pre-bakeable). We decouple *what to keep* (selection)
  from *where to place it* (position) from *which query scores it*, and show the
  deployable, query-free part is position-preserving reuse (+ optional self-saliency).

## Files

- Diagnostic runners (gen dump + `_degen`; 35 adds the `fresh_nodup` arm):
  `scripts/34_a3b_diag.py`, `scripts/35_q27_diag.py`
- Fixed coverage runners (sink-dup removed): `scripts/36_a3b_fix.py`,
  `scripts/37_q27_fix.py`
- Clean results: `data/a3b_cov_fix.s*.json` (n=231), `data/q27_cov_fix.s*.json` (n=234)
- Diagnostics: `data/q27_diag.s*.json` (n=231, 3 arms), `data/a3b_diag.s*.json`
