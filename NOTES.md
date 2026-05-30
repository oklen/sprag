# sprag — design notes & status

> Last updated: 2026-05-29 (RGB benchmark — cache K/V net-negative on real
> passages (§5u); per-subspace α probe (§5v); K-vs-V decomposition on MK —
> splice cost is cache→assembly DRIFT (§5w); BUT RGB validation falsifies it
> (§5x) — drift/coherence/K-vs-V are synthetic-MK artifacts, on real passages
> cache K/V is monotonically harmful regardless of construction. Robust
> conclusion: sprag = short-assembly format + sink; K/V splice is the
> incremental/negative piece. Fixed symmetric anchor (§5y): on MK ≈ doc-sink
> anchor (single front anchor only de-drifts the top-1 chunk; E(0.5)=60/60),
> but RGB validation (§5y-RGB) shows it's monotonic-harmful AND uniformly
> WORSE than standard/anchor (raw 75.0/E 68.7/α1 64.0) — the bare endoftext
> sink underperforms a content-bearing one even on the fresh path. Verdict:
> cleaner construction, not a valuable one). Linear-attn state blend (§5z):
> the 18 GatedDeltaNet layers (always fresh until now) — caching+blending the
> recurrent state is free-at-best/never-better (MK norm-matched α0.5=54 vs
> fresh 58) and α=1 collapses (17/60); no compute skip either. Splice
> conclusion now spans BOTH attention families. full−pos+fresh delta cache
> (§5aa): residual-add on K/V, linear, or both is monotone-harmful, none beats
> fresh, literal α=1 collapses (K/V 6, linear 15, both 3 / 60); fails on
> magnitude not position (fresh is already complete, residual over-drives it).

## 1. Why this exists

We want to test whether **ReAttention** (display-position decoupling via
unrotated KV cache + Inverse RoPE) combined with **MAGS** (manifold-guided
residual-stream steering) can mitigate two failure modes of long-context
LLM inference for RAG:

- **Explicit position drift** — when a chunk's tokens, originally at
  document position `A`, are spliced into an assembled prompt at
  position `B`, RoPE-rotated K vectors no longer line up.
- **Implicit semantic drift** — even after explicit positions are
  fixed, the residual stream during decode wanders into a direction
  that the model has learned associates with "wrong context" or
  hallucination.

Target: **Qwen3.5-0.8B**, which is a hybrid:

- 24 layers; every 4th layer (3, 7, 11, 15, 19, 23) is a standard
  full-attention block (GQA 8q/2kv, head_dim=256).
- The other 18 layers are **Gated DeltaNet** linear-attention with
  causal Conv1D + recurrent state.
- Full-attn uses **MRoPE** (3-axis interleaved) with
  `partial_rotary_factor=0.25` — only 64 of 256 head dims are rotated.
  In pure-text mode the 3 axes share position ids, so MRoPE degenerates
  to standard 1D RoPE on those 64 dims. *(Verified numerically:
  `tests/test_rope.py` shows 0 error.)*

## 2. Three-pronged solution

### 2.1 Full-attn layers — Inverse RoPE

`K_target = R_B · K_raw = R_{B-A} · (R_A · K_raw) = R_{B-A} · K_cached`.

We cache `K_cached` (post `k_norm`, post-RoPE at original position A) and
`V`. At query time, for each retrieved chunk we apply one rotation
`R_{B-A}` to the rotated slice of `K_cached`. This is implemented as a
monkey-patch on each of the 6 full-attn `forward()`s that **splices**
the rotated K/V into `key_states` and `value_states` after the regular
RoPE step, before the cache update and attention computation.

See `src/sprag/rope.py` (the rotation math) and `src/sprag/assemble.py`
(the patched forward).

### 2.2 Linear-attn layers — LegoLink

**v1 (current)**: do nothing special. The assembled prompt
(prefix + K retrieved chunks + query) is short — typically ≤ 1.5K
tokens — so the 18 linear-attn layers just process it normally as an
O(N) sweep.

**v2 (to do on GPU)**: cache per-chunk `(G_c, M_c)` decomposition.
For Gated DeltaNet, `S_end = G_c · S_init + M_c` is exact, where
`G_c = Π γ_t` and `M_c = Σ_t (Π_{s>t} γ_s) β_t k_t v_t^T`. With this we
can prepend a chunk to any prefix state in O(1) state stitching plus
optional ~32-token re-forward to smooth the Conv1D boundary (the conv
kernel size is 4).

The transformers `Qwen3_5GatedDeltaNet.forward` already accepts
`initial_state` and returns `last_recurrent_state`, so v2 is mostly
about engineering the (G_c, M_c) capture during chunk preprocessing.

### 2.3 All layers — MAGS

Hook the residual-stream output of layers **11 / 15 / 19** with a
PyTorch `register_forward_hook`. Offline:

```
mu_c = mean of T+ trajectory residuals
D    = T- residuals minus mu_c
B    = top-k right singular vectors of D     (k=4)
tau  = 95th percentile of ||B(P - mu_c)|| on T+
```

Online (decode), per new token:

```
d = || B (a - mu_c) ||
if d > tau:
    a ← a - alpha * Bᵀ B (a - mu_c)
```

`alpha=1.0` is full orthogonal projection out of the error subspace.

See `src/sprag/mags/{calibrate,intervene}.py`.

## 3. What's verified

| Test | Result |
|---|---|
| MRoPE in text-only == 1D RoPE on rotated dims | abs error 0 |
| `apply_rope` matches transformers' `apply_rotary_pos_emb` | abs error 0 |
| Inverse-RoPE shift: `R_{B-A} · R_A · x = R_B · x` | rel error 6e-5 (fp32 trig floor) |
| Identity-splice (cache replays original positions) | bit-exact 0 difference vs unpatched forward |
| End-to-end smoke (Octavia/forty-two needle) | baseline ✓, ReAttention ✓ |

## 4. 4K NIAH measurements (CPU, 5 cases)

Hardware: 2-core AMD EPYC slice, 7.8 GB RAM, no GPU.

| Mode | acc | timing per case |
|---|---|---|
| baseline (4K inline) | 3/5 = 60 % | 30–49 s prefill+gen |
| ReAttention top-3 (256-tok chunks, 768-tok assembly) | 3/5 = 60 % | 65 s cache build, 9 s query |
| Full (ReAttention + MAGS) | 3/5 = 60 % | 9 s query (after cache) |

**Failure mode breakdown:**

- Case 1 — both modes output "93" instead of the needle's correct number; baseline can't extract the right one from noise, ReAttention's top-3 missed the needle chunk entirely.
- Case 3 — same retrieval-miss pattern for ReAttention; baseline misses for a different reason (formatting).
- Case 4 — baseline says "93", ReAttention says "ninety-three" (it copies the chunk verbatim). This is a real ReAttention win.

**MAGS fire-rate audit on cases 3 + 4** (see `scripts/05_inspect_mags_fire.py`):

| layer | τ | case 3 fired | case 4 fired |
|---|---|---|---|
| 11 | 0.378 | 15/23 (65 %) | **23/23 (100 %)** |
| 15 | 0.693 | 16/23 (70 %) | 9/23 (39 %) |
| 19 | 2.613 | 5/23 (22 %) | 6/23 (26 %) |

Conclusion: MAGS hook plumbing works (it fires; distances are in a
reasonable range) **but the calibration is degenerate** — only 8
T+/T- pairs is too few for a meaningful SVD, and layer 11 fires on
100 % of *correct* generations. To make MAGS earn its keep we need
50+ calibration pairs on longer contexts where the model genuinely
drifts.

## 4b. 16K & 32K NIAH measurements (Tesla T4, fp16, SDPA mem-eff)

Hardware: Tesla T4, 15 GB, compute-cap 7.5. Single-needle NIAH.

| Context | Mode | acc | timing per case |
|---|---|---|---|
| 16K | baseline | 7/10 = 70 % | 5–8 s prefill+gen |
| 16K | ReAttention top-5 (256-tok chunks, 1280-tok assembly) | 7/10 = 70 % | 15 s cache + 2 s query |
| 32K | baseline | 3/5 = 60 % | 12–13 s prefill+gen |
| 32K | ReAttention top-5 | 3/5 = 60 % | 33 s cache + 2 s query |

Amortization break-even: at 16K, ReAttention pays back after ≈ 3
queries on the same doc (15 s / (8 s − 2 s) ≈ 2.5). At 32K, ≈ 3
queries (33 / (13 − 2)). For RAG workloads with many queries per
doc this is a big win; for single-shot Q-over-long-doc the cache
cost is dominant.

Failure modes still split between retrieval miss and "model can't
extract from the chunk even when it's there." The latter looks
like the case MAGS should help with — see §5b.

## 5a. Critical fix (GPU-only)
The reattn assembled prompt initially had no separator between
context chunks and the `Q:` line, which caused the model to echo
the query without using context. Adding `\n\n` before the query
recovered 4/10 cases at 16K. The NIAH driver passes
`"\n\nQ: ..."` to `runner.run()`; the runner itself is
format-agnostic.

## 5b. MAGS 16K calibration & full-mode eval

Refit MAGS on 60 NIAH-style cases at 16K (`scripts/04_calibrate_mags.py
--cases data/niah/niah_16k_calib.jsonl --n_calib 60 --k_svd 4 --n_wrong 3`).
59/60 cases yielded an oracle chunk (one case's needle straddled a chunk
boundary and was skipped).

| layer | τ (95th-pctile of T+ distances) |
|---|---|
| 11 | 0.296 |
| 15 | 0.881 |
| 19 | 2.607 |

These taus are close to the 8-pair CPU calibration's, but now backed by a
real (59, 1024) SVD per layer.

| Mode | 16K acc |
|---|---|
| baseline | 7/10 = 70 % |
| reattn | 7/10 = 70 % |
| full (reattn + MAGS) | 7/10 = 70 % |

MAGS neither helps nor hurts: it preserves accuracy on all 7 cases where
reattn already succeeds, and doesn't recover the 3 failures (which are
retrieval misses — the needle chunk isn't in the top-5).

Fire-rate audit on cases 6 (fail), 8 (success), 9 (success):

| layer | τ | case 6 fired | case 8 fired | case 9 fired |
|---|---|---|---|---|
| 11 | 0.296 | 21/23 (91 %) | 13/23 (57 %) | 18/23 (78 %) |
| 15 | 0.881 | 15/23 (65 %) | 7/23 (30 %) | 6/23 (26 %) |
| 19 | 2.607 | 3/23 (13 %) | 6/23 (26 %) | 4/23 (17 %) |

Layer 11 still fires on a majority of *correct* generations, so its τ is
not selective enough — the orthogonal projection at layer 11 is operating
on most tokens of most generations and the net direction it removes is
small enough that the answer survives. Layer 19 looks discriminating
shape-wise but fires too rarely to drive aggregate accuracy.

The bigger issue (predicted in §8) is exposed: bottom-K-by-cosine "wrong"
chunks aren't *plausibly mis-retrievable* enough to teach the SVD a
useful error direction for the failure mode MAGS actually targets
(retrieval-was-OK-but-model-still-drifted). For the next iteration, T-
should be drawn from *nearly-correct* retrievals — chunks whose cosine is
just below the oracle — or from synthetic perturbations of the oracle
itself.

## 5e. Multi-Key NIAH suite (8K, 10 cases × 6 needles)

Generated by `scripts/data/gen_mk_suite.py`; evaluated by
`scripts/07_run_mk_niah.py` with the numeric-aware scorer
(`"42" ≡ "forty-two"`, etc.) and a 3-way classifier — correct /
distractor (output matches a sibling needle's answer in the same doc) /
other (degenerate or unrelated). 60 queries total.

| Mode | correct | distractor | other | per-q |
|---|---|---|---|---|
| baseline (full prompt) | 57/60 | 2 | 1 | 2.81 s |
| reattn top_k=3 | 24/60 | 2 | 34 | 1.68 s |
| reattn top_k=6 | 39/60 | 6 | 15 | 1.86 s |

Per-template breakdown of `reattn_k6`:

| Template | correct |
|---|---|
| vault (number) | 20/22 |
| secret-keeper | 10/10 |
| bookshop (city → street) | 9/28 |

Bookshop is the failure case. Vault and secret-keeper recover at
top_k=6; bookshop barely budges. The dominant failure mode for bookshop
is `other` — the assembled prompt collapses into `\nA:\nA:\nA:...`
degenerate output. The query embedding ("Where is the best quiet
bookshop in X") doesn't discriminate well by city — retrieval keeps
pulling the same bookshop chunks regardless of which city was asked,
and when the assembled context contains three or four bookshop needles
the model goes off the rails rather than picking one. A real
"distractor" hallucination (choosing the wrong bookshop) appears 6/60
times under k=6, but the silent degenerate-output failure is more
common.

This is the MAGS-worthy failure mode: retrieval at least *included*
the right needle in many of these cases (e.g. q4 in case 0: gold
"Marigold", retrieved chunks include the Marigold-Kyoto needle, but
the model output "Mulberry"). The CPU-era MAGS calibration won't help
here — see §5b. Next: recalibrate MAGS with T- drawn from these
*plausibly mis-retrievable* bookshop cases, not from low-cosine random
chunks.

## 5f. Perf cleanup (post-MK)

Three changes after the MK run, none touching numerics:

1. **chunk_cache single-write** (`src/sprag/chunk_cache.py`). The
   old path saved each chunk's safetensors twice — once for K/V,
   then reloaded + re-saved after the Jina embedding batch. Now
   tensors are assembled in memory and written once. Cold-start
   case-0 cache build at 8K: 21.7 s → 8.57 s.
2. **Eager chunk load on GPU** (`src/sprag/runner.py`). Disk-load of
   chunk tensors now happens once at `SpragRunner.__init__`,
   memoised by `(cache_dir, device, dtype)` and shared between
   runners pointing at the same cache (e.g. top_k=3 and top_k=6 in
   the MK eval). K/V live on GPU in model dtype, so the per-splice
   `.to(device).to(dtype)` calls in `patched_full_attn` become no-ops.
   `chunk_cache.build_chunk_cache` now also invalidates the memo for
   any rebuilt cache dir.
3. **TF32 + matmul precision** (`src/sprag/loader.py`).
   `torch.set_float32_matmul_precision("high")` plus
   `torch.backends.{cuda.matmul,cudnn}.allow_tf32 = True` — only
   applies to the residual fp32 paths (RoPE angles, some norm
   reductions). Free speedup on Ampere+; on T4 it is largely a no-op
   because TF32 needs cap ≥ 8.0, but the calls don't hurt.

Per-query latency (baseline ≈ 3 s, reattn ≈ 2 s) was unchanged within
noise — the prior bottleneck was `model.generate()` (prefill + 32-token
decode), not the per-query I/O these changes target. The amortization
break-even (§5d) is unaffected: cache-build is cheaper, per-query is
the same. For per-query wins the remaining levers are
prefix-KV-cached baseline (the fair comparison for true RAG workloads),
shorter `max_new_tokens`, or speculative decoding — all deferred.

## 5g. MAGS recalibration on MK — negative result

Two attempts to re-fit MAGS against the bookshop failure mode from §5e.
Both use the MK suite as the source of (T+, T-) pairs (much cleaner
than the §5b CPU-era 8-pair bottom-K-cosine T-).

**Attempt 1 — sibling-template T-** (`scripts/08_calibrate_mags_mk.py`).
For each query: T+ = oracle-chunk-only assembly; T- = same-template
*sibling-needle* chunks only, no gold. 49 pairs.

```
τ:  layer 11 = 0.418   (was 0.296 in §5b)
    layer 15 = 1.137
    layer 19 = 3.356
```

Fire-rate audit on cases 0–1 (6 queries each):
- Layer 11 fires 4–9/23 tokens on bookshop queries (both correct AND
  failed), 0–3/23 on vault/secret correct queries.
- Layer 15/19 essentially silent.

So τ became selective on *template family*, not on *failure*. The
SVD direction encodes "I am answering a bookshop question," which is
mostly innocuous.

**Attempt 2 — harvest from eval failures**
(`scripts/10_calibrate_mags_harvest.py`). T+ = queries where the
runner answered correctly AND the gold-needle chunk was in retrieved;
T- = queries where the gold chunk was retrieved but the model
answered with a distractor or degenerated. 39 T+ / 20 T-. This is the
"plausibly mis-retrievable" T- §8 called for: retrieval gave the
model the right needle, the model still drifted.

```
τ:  layer 11 = 0.375
    layer 15 = 1.298
    layer 19 = 3.208
```

Re-eval on the 10-case MK suite (60 queries, top_k=6):

| Mode | correct | distractor | other |
|---|---|---|---|
| baseline | 57/60 | 2 | 1 |
| reattn_k6 | 39/60 | 6 | 15 |
| **full_k6 (harvest MAGS)** | **38/60** | 7 | 15 |

Per template (full_k6 vs reattn_k6): vault 20/22 vs 20/22, secret
10/10 vs 10/10, **bookshop 8/28 vs 9/28**. MAGS turned exactly one
case from correct → distractor; everything else identical.

**Why MAGS doesn't fix bookshop, even with clean T-.** The bookshop
failure isn't "the residual drifts in a uniform direction across
failures." Two distinct failure shapes appear:

1. *Degenerate* (≈13/19 of failures): the assembled prompt with 3–4
   bookshop needles makes the model produce `\nA:\nA:\nA:...` — a
   complete refusal to commit to any answer. SVD-projecting a
   subspace out of a runaway-attention residual doesn't recover the
   answer; the model already lost the question.
2. *Mis-binding* (≈6/19): "Where in Kyoto?" → "Mulberry" (Tallinn's
   street). The model selected the wrong needle's binding. Fixing this
   requires moving the residual from "I attend to needle Y" to "I
   attend to needle X" — a binding-specific transform, not a uniform
   linear subtraction.

A single shared subspace B with a single threshold τ has the wrong
shape for either mode. SVD on (correct, drifted) residuals gives the
*average* drift, which is small and not aligned with any specific
mis-binding direction.

**What would actually move the needle (deferred):**

- Per-template (or per-template-and-position) MAGS subspaces.
- Intervene at attention scores rather than residual — the binding
  decision happens there.
- Inspect attention maps on failures to confirm the binding
  hypothesis.
- Or accept that for this failure mode, retrieval (chunk_repr
  ablation §7.5) is the right lever, not residual intervention.

## 5h. Oracle retrieval on MK — which open problem is the real one

To separate "retrieval picked the wrong chunks" from "splice itself is
weak when multiple needles compete," we ran an oracle eval
(`scripts/11_oracle_mk.py`) that bypasses Jina and assembles only the
chunk known to contain each query's gold needle.

`oracle_k1`: assembly = [gold_chunk] only.
`oracle_k3`: assembly = [gold_chunk, sibling_needle_1, sibling_needle_2]
             (gold always first, distractors of same template after).

Same 10-case MK suite (60 queries):

| Mode | total | vault | secret | bookshop |
|---|---|---|---|---|
| baseline (full) | 57/60 | 22/22 | 10/10 | 25/28 |
| reattn_k6 (Jina) | 39/60 | 20/22 | 10/10 | 9/28 |
| **oracle_k1** | 18/60 | 6/22 | 9/10 | 3/28 |
| **oracle_k3** | 41/60 | 15/22 | 10/10 | **16/28** |

Three things drop out:

1. **Bookshop: retrieval discrimination is the dominant bottleneck.**
   oracle_k3 with the same splice machinery jumps from 9/28 → 16/28
   on bookshop just by re-ordering Jina's top-6 into "gold first +
   2 sibling distractors." Jina at top-6 *did* include the gold
   chunk ~99 % of the time (see §5g harvest stats), but it ranked
   the gold chunk indistinguishably from sibling-bookshop chunks —
   the model in that 6-chunk soup picks the wrong one too often.
   The bookshop binding error in §5g exists, but it is much less
   damaging when the gold chunk is first and competing needles are
   capped at 2.

2. **The splice is still imperfect (bookshop 16/28, not 25/28).**
   Oracle gold-first with only 2 distractors still loses 9 of the
   28 bookshop cases — most as `other` (degenerate decode). The
   spliced K/V at large Inverse-RoPE deltas (e.g. chunk 22 →
   position 0, delta ≈ 5600) plus a multi-needle assembly is enough
   to push the model into a degenerate state on some queries. This
   is the empirical signature of open problem 4 (Inverse-RoPE K is
   exact on rotation, but "semantically correct K for the new
   position" is not validated).

3. **Vault gets *worse* under oracle_k3** (15/22 vs reattn_k6 20/22).
   Reason: Jina at top-6 pulled 6 chunks, most of which are filler
   (non-needle) text — the model just sees one or two vault needles
   surrounded by neutral context. oracle_k3 instead has 3 vault
   needles packed back-to-back, which intensifies the multi-needle
   confusion. So "more chunks" actually helps vault by *diluting*
   competing needles. The bookshop fix and the vault fix point in
   opposite directions.

   This makes a single retrieval policy hard: for bookshop you want
   "very tight, gold-first, fewer competing needles"; for vault you
   want "broader top-k so competing needles are dispersed in
   filler." Per-template retrieval is one fix; another is reranking
   Jina's top-k by keyword match on the disambiguating field
   (`city`/`vault`/`name`) before assembly.

**`oracle_k1` collapse** (18/60) is mostly a context-length artifact —
a 256-token assembly is too short for this model + Q&A format; it
degenerates regardless of splice quality. We tag this as confounded
and do not read splice quality off it.

What this rules out / in for the open problems in §8:

- §8 *chunk_repr space mismatch* (open problem 3): **confirmed the
  dominant lever for bookshop**. The chunk_repr ablation (use Qwen's
  own `repr_mean_last`) is now the highest-EV experiment.
- §8 *Inverse-RoPE semantic correctness* (open problem 4): **also
  contributes**, but is second-order under good retrieval — bookshop
  9/28 → 16/28 from a retrieval reorder alone.
- §8 *MAGS calibration*: confirmed dead-end for bookshop binding
  errors — even with perfect retrieval the residue is degenerate
  decode, not the kind of uniform-direction drift MAGS could fix.

## 5i. StreamingLLM sink prepend — big bookshop win, small vault regression

Reading §5h's "degenerate decode is the residual failure mode on
bookshop" against the Xiao et al. *Attention Sink* result, we tried a
no-retrain intervention: prepend the **first M=4 tokens of the doc** as
a sticky global sink at b=0, and strip the **first S=4 tokens off every
retrieved chunk** before splicing (each placement's a_start is bumped
by S, and `cached[li]` is sliced `[:, S:, :]`). ReAttention's existing
per-placement RoPE rebase handles the rotation for both — the sink has
delta=0 (no rotation), each stripped chunk shifts by `b_start - (a+S)`.
No model changes, no extra forward.

Run: `scripts/12_sink_mk.py --suite data/mk/suite_8k --M 4 --S 4`,
same 10-case / 60-query MK suite, four modes side-by-side:

| Mode | total | vault | secret | bookshop | other |
|---|---|---|---|---|---|
| oracle_k3 (control, §5h) | 41/60 | 15/22 | 10/10 | 16/28 | 15 |
| **sink_oracle_k3** | **47/60** | 16/22 | 10/10 | **21/28** | 11 |
| reattn_k6 (control, §5e) | 39/60 | 20/22 | 10/10 | 9/28 | 15 |
| **sink_k6** | **43/60** | 17/22 | 10/10 | **16/28** | 8 |
| baseline (full prompt) | 57/60 | 22/22 | 10/10 | 25/28 | — |

Three things drop out, in order of size:

1. **The sink rescues degenerate-decode failures.** Across all modes,
   the `other` (degenerate "\\nA:\\nA:..." collapse) bucket falls
   15 → 11 under oracle and 15 → 8 under retrieval. The
   StreamingLLM hypothesis was exactly this — when no token in scope
   is positioned at the absolute first few RoPE indices, attention
   has nowhere to dump its "I have nothing to say" mass, and decode
   becomes pathological. Restoring a sticky [0..4) anchor fixes a
   majority of those cases.

2. **Bookshop closes most of the gap.** Plain ReAttention was stuck
   at 9/28 (§5e); oracle reorder lifted to 16/28 (§5h); now
   *retrieval* with sink reaches the previous *oracle ceiling*
   (sink_k6 = 16/28), and sink + oracle pushes past it (21/28). The
   bookshop wins come almost entirely from the degenerate bucket
   collapsing, not from binding errors changing — the
   `distractor` bucket actually grows on retrieval (6 → 9) because
   some queries that previously degenerated now generate the
   sibling-needle answer instead. Net is still strongly positive.

3. **Vault regresses 20/22 → 17/22 under retrieval.** Vault was
   already saturated by reattn_k6's broad top-6 (where competing
   needles get diluted in filler chunks); adding a sink and trimming
   the first 4 tokens of each retrieved chunk costs 3 cases.
   Inspection of the lost cases: in two, sink_k6 produces a distractor
   ("vault Beta is forty-two" when asked about Alpha) where reattn_k6
   produced the correct answer. We hypothesise the stripped first 4
   tokens sometimes contain part of the "vault X is …" preamble, so
   the binding cue is weaker. This is an `S` tuning lever — a
   follow-up sweep over `S ∈ {0, 2, 4}` should localise where the
   tradeoff sits.

What this changes about the open problems:

- §8 *Inverse-RoPE semantic correctness* (open problem 4): The
  degenerate-decode residue we attributed to "Inverse-RoPE K is
  semantically wrong at new positions" is largely explained by an
  unrelated mechanism — *no absolute-position-0 anchor in scope*.
  Once a sink is restored, much of the residue resolves. Open
  problem 4 still exists but is smaller than §5h estimated.
- §8 *chunk_repr space mismatch* (open problem 3): Still the
  dominant lever for the *remaining* bookshop gap (sink_k6 16/28 vs
  baseline 25/28). Worth pursuing.
- **Strip length S is not a useful lever.** Sweep over S ∈ {0, 2, 4}
  with M=4 fixed:

  | S | sink_oracle_k3 | sink_k6 | bookshop (k3/k6) | vault (k3/k6) |
  |---|---|---|---|---|
  | 0 (sink-only, no strip) | 46/60 | **44/60** | 20 / 17 | 16 / **17** |
  | 2 | 44/60 | 41/60 | 19 / 16 | 15 / 15 |
  | 4 (canonical) | **47/60** | 43/60 | **21** / 16 | 16 / **17** |

  Vault is non-monotonic in S (17 / 15 / 17), so the regression isn't
  driven by strip width — it's the *sink itself* absorbing some
  attention that previously went to the vault keyword, plus
  greedy-decode variance. Bookshop is roughly flat across S=0..4.

  This rules out the StreamingLLM rationale for stripping. The
  original paper strips because in a *sliding-window* setup each
  kept token would otherwise inherit a token's worth of "this is a
  sink" attention from its sliding position; in our setup ReAttention
  already positions each chunk wherever we want, so there's no
  conflict to resolve by trimming.

  **Recommended default: M=4, S=0** — sink-only prepend, no strip.
  Gives the same bookshop win, costs no chunk information, free at
  runtime (+4 tokens to each prefill).

## 5j. Splice divergence — direct K diagnosis

§5i raised a hard question: oracle splice gets 47/60 but the *exact same
chunks* fed as raw text get 58/60. An 11/60 gap on identical content
must be coming from inside the splice mechanism. We measured it
directly.

For each of the 60 MK queries, during a real `sink_oracle_k3` prefill
(M=4 sink + gold + 2 sibling chunks), we instrumented each of the 6
full-attn layers to capture three K tensors at every chunk position:

  - `K_fresh`    — what the model just computed from the assembled
                   hidden states (= what splice is about to overwrite)
  - `K_shifted`  — `shift_rope(K_cached, delta=b−a)` (= what splice writes)
  - `K_cached`   — the raw cached K at original position a (no rotation)

Per-layer × role aggregates (means over 60 queries, all 6 full-attn layers):

| layer | sink (Δ=0) cos | sib0 (Δ≈-1191) cos | sib1 (Δ≈-2129) cos | gold (Δ≈-3938) cos |
|---|---|---|---|---|
| 3  | 1.0000 | 0.970 | 0.972 | 0.886 |
| 7  | 1.0000 | 0.888 | 0.887 | 0.740 |
| 11 | 1.0000 | 0.745 | 0.768 | **0.512** |
| 15 | 1.0000 | 0.751 | 0.773 | 0.558 |
| 19 | 1.0000 | 0.811 | 0.820 | 0.670 |
| 23 | 1.0000 | 0.806 | 0.812 | 0.662 |

(rel_shifted at L11 gold = 1.00 — magnitudes also off by 100 %.)

Three results lock in:

1. **The RoPE math is correct.** The sink placement (Δ=0) gives
   rel_err ≈ 1e-3 and cos ≈ 1.0000 to four decimals across all 6
   layers. So `shift_rope` and the cached K/V are numerically
   consistent with what the model would compute fresh at the same
   position with the same hidden state.

2. **The drift dominates the rotation.** Compare `rel_shifted` (K with
   RoPE delta-rotation) to `rel_cached` (K without rotation, still at
   original position): `rel_shifted` is consistently *smaller*, by
   ~5–30 %. So the RoPE delta-rotation *helps* — but the residual gap
   is the bigger fraction. At layer 11 gold: `rel_shifted = 1.00` vs
   `rel_cached = 1.04` — rotation closed 4 % of the gap, the other
   96 % is hidden-state drift.

3. **Drift grows with depth and with |Δ|.** Layer 11/15 are worst
   (cos ≈ 0.5–0.55 for gold), surface layers 3/7 better
   (cos ≈ 0.74–0.89). Within a layer, sib0/sib1 (smaller |Δ|) drift
   less than gold (larger |Δ|). Both consistent with: deeper layers'
   hidden states accumulate more dependence on surrounding context;
   chunks moved further from their original positions diverge more.

### What this means

The cached K at full-attn layer L for position a is

  `K_cached = RoPE(k_norm(k_proj(h_L_a^orig)), a)`

where `h_L_a^orig` is the hidden state at layer L, position a, *as
produced by the full-haystack forward pass*. The splice writes this
into the assembled prefill at position b after RoPE-rotating by Δ.
But the model, attending from query positions in the new context,
would naturally compute

  `K_fresh = RoPE(k_norm(k_proj(h_L_b^new)), b)`

where `h_L_b^new` is the hidden state at layer L, position b, *in the
assembled context*. These two hidden states are not the same — and
they can't be, because every layer below L attends over a different
set of surrounding tokens. The cached K is "stale" not in terms of
RoPE rotation but in terms of the very residual stream that produced
it.

This is the cleanest characterisation we now have of open problem 4
("Inverse-RoPE semantic correctness"). It is not a *math bug* — the
formula is right and we can prove it numerically. It is a *premise
bug* — ReAttention assumes K is a function of (token, position), but
in this model K is a function of (token, position, lower-layer
context). When you change the context, K drifts even if you keep token
and position fixed. Whatever fraction of attention's job is "score
which earlier token is relevant", that fraction is being done with
K vectors that are 50–80 % wrong.

### Implications for the path forward

- A pure cache-and-rotate scheme cannot close this 11/60 gap on
  this model. There is no per-chunk preprocessing of the cache that
  recovers `h_L_b^new` from `h_L_a^orig` — it would require knowing
  the assembled context at cache-build time.
- Recovery options worth scoping:
  - **Partial re-prefill on the top K layers** — tested in §5k below
    and **fails**. The U-shape there shows partial splice is worse
    than either extreme; the K dependency is genuinely cross-layer
    and can't be cherry-picked.
  - **Anchor-conditioned cache.** Cache K/V with a short surrounding
    context window (~32 tokens before and after each chunk) included
    in the cache-build forward, so `h_L_a^cache` is closer to
    `h_L_b^new` for typical assembly shapes.
  - **Hidden-state caching instead of K/V caching.** Cache `h_L`
    directly; at query time apply k_norm/k_proj/RoPE for the new
    position. Same drift problem in `h_L` itself, but with a single
    layer of indirection removed.
- The good news for the project's value proposition: §5i shows the
  sink prepend rescues most degenerate-decode failures *despite*
  this drift. So even with the drift baked in, the system is usable;
  the open question is how to recover the 11/60 cases that the drift
  costs.

Diagnostic data: `data/diag/splice_div.json` (rows of per-query,
per-layer, per-role L2 / cosine measurements). Script:
`scripts/13_diagnose_splice.py`.

## 5k. Partial re-prefill sweep — the U-shape

§5j said: K drift is concentrated at deep layers (L11/15 worst,
cos ≈ 0.5 vs fresh). Natural hypothesis: splice K/V only at the
shallow layers where the cache is faithful, let the deep layers
re-compute K/V over the assembled context. Maybe this gets us close
to raw_oracle_k3 (58/60) without giving up the cache entirely.

`patched_full_attn(splice_layers=...)` now takes a subset of
FULL_ATTN_LAYERS; unlisted layers run normally. Sweep over the
nested chain:

| splice_layers | total | vault | secret | bookshop | other |
|---|---|---|---|---|---|
| [] (= raw_oracle_k3, §5i) | **58/60** | 22/22 | 10/10 | 26/28 | 1 |
| [3] | 51/60 | 21/22 | 8/10 | 22/28 | 6 |
| [3, 7] | 39/60 | 16/22 | 9/10 | 14/28 | 5 |
| **[3, 7, 11]** | **29/60** | 10/22 | 9/10 | 10/28 | 11 |
| [3, 7, 11, 15] | 35/60 | 15/22 | 8/10 | 12/28 | 13 |
| [3, 7, 11, 15, 19] | 49/60 | 18/22 | 10/10 | 21/28 | 8 |
| [3, 7, 11, 15, 19, 23] (full, = sink_oracle_k3) | 47/60 | 16/22 | 10/10 | 21/28 | 11 |

The U-shape is unmistakable. Both extremes work; the middle
collapses. [3,7,11] hits 29/60 — *worse than baseline-minus-30,
worse than every other configuration*.

### Why partial breaks

Splicing K/V at layer L means: at L, the attention attends using
K_cached (from full-haystack hidden state). The output of layer L
then propagates through linear-attn layers L+1, L+2, L+3 to the
next full-attn layer L'. At L', if we splice again, the chain is
self-consistent — every full-attn layer attends with full-haystack
patterns, and the linear-attn layers in between propagate states
that, while not identical to original, are at least driven by the
same "the model is attending to original context" signal.

If we *don't* splice at L', then L' computes K fresh from h that
was already distorted by upstream splicing. The fresh K reflects
"a layer trained to expect natural propagation, fed an unnaturally
distorted hidden state." The deeper this distortion-then-natural
transition is, the worse the model behaves — until the chain is
short enough that natural propagation reasserts (last row).

The [3,7,11] minimum sits right at the layer 11 transition, where:
  - L3 splice + L7 splice has already deformed h_8 through h_10
    (linear-attn layers running on assembled-context-with-spliced-L3-L7)
  - At L11 we splice again, plant cached K from full-haystack context
    on top of this already-deformed h_11
  - L12+ then run natural over this maximally-confused state

Splicing at all 6 layers (47/60) or splicing at none (58/60) both
let the model run in a self-consistent regime. Anything between is
mismatched.

### What this rules in / out

- **The "cache shallow, re-prefill deep" plan in §5j is dead.** It
  would have been the cleanest fix; it isn't available.
- **The K drift in §5j is a *symptom*, not the disease.** The disease
  is that K depends on h, h depends on the entire upstream layer
  chain, and the chain can't be cleanly partitioned. Any cache that
  freezes K/V at one layer is implicitly freezing assumptions about
  every layer below it.
- **The cache-and-splice premise is structurally limited on this
  model.** ReAttention can match raw-text at best when the chunks'
  original positions are very close to their assembled positions
  (Δ small, drift small). For multi-needle long-doc retrieval where
  Δ is in the thousands, you pay ~11/60 (~18 %) for the splice.
- **Remaining levers** are all about reducing the h_a vs h_b gap at
  cache-build time, not about how to use the cache at query time:
  - Anchor-conditioned cache (~32 tok pre/post in cache forward).
  - Hidden-state caching (cache h_L, recompute K/V at query time
    per assembled position — h_L itself drifts but the projection
    step is at least faithful).
  - ~~Cache **only V**, recompute K fresh~~ — tested in §5l below and
    **catastrophic** (5/60). K and V have to be coherent (same
    forward pass); separating them is the worst possible setting.

## 5l. K-only / V-only splice — K-V coherence is non-negotiable

Hypothesis (now rejected): "K is what's drifting badly per §5j, so
recompute K fresh and keep cached V. Attention pattern correct, content
correct — should give us close to raw_oracle_k3 with the cache still
amortising the cache-build forward."

Reality (60 queries, full MK suite):

| Mode | total | vault | secret | bookshop | other |
|---|---|---|---|---|---|
| raw_oracle_k3 (no splice) | 58/60 | 22/22 | 10/10 | 26/28 | 1 |
| sink_oracle_k3 (full KV splice) | 47/60 | 16/22 | 10/10 | 21/28 | 11 |
| **k_only_oracle_k3** | 29/60 | 8/22 | 10/10 | 11/28 | 17 |
| **v_only_oracle_k3** | **5/60** | 0/22 | 5/10 | 0/28 | 40 |

V-only collapses entirely on vault (0/22) and bookshop (0/28); 40 of
60 outputs are degenerate. K-only ≈ the §5k U-shape bottom.

### Why decoupling K from V is worse than either extreme

In a normal forward pass, K and V at any position arise from a single
`h` via `K = RoPE(k_norm(k_proj(h)), pos)` and `V = v_proj(h)`. They
are coherent: every position's V is "the content the model would
attend to if it scored that position highly via Q × K^T".

- **V-only splice**: Q × K_fresh^T computes attention scores over the
  *current* assembled context. The scores reflect "what looks
  relevant in this short-context residual stream." Then the model
  pulls V_cached at those positions — but V_cached at position b is
  not the content at position b in the current assembly. It is the
  content that was at position a in the *original* full-haystack
  context. So the model is, for every attention head: choosing
  attention slots via one context, retrieving content from another.
  Decoder behavior decays into mode collapse (`\nA:\nA:\nA:...`).

- **K-only splice**: Q × K_cached^T scores based on original-context
  K patterns — already wrong (K_cached at position b was trained for
  Q from original context, not current). The model attends to weird
  positions, but at least V_fresh at those positions is real content
  of the assembled sequence. Survivable but bad (29/60).

- **Full KV splice**: K_cached + V_cached are coherent (same forward
  pass). The attention scores are wrong for the current context, but
  the (K, V) pairs are internally consistent. The model effectively
  re-experiences a "trim of the original forward pass." 47/60.

- **No splice (raw)**: Everything fresh, fully coherent in current
  context. 58/60.

Conclusion: ReAttention's cache mechanism is structurally bound to
splice both K and V together, per layer, in lockstep. There is no
"compute the cheap part fresh, splice only the expensive part" trick
on this model.

### What's left for the project

After §5j / §5k / §5l, the cache-and-splice idea has a firm ceiling
on this model:

  cache+splice ceiling ≈ 47/60     (`sink_oracle_k3`, M=4, S=0/4)
  no-cache (raw) ceiling ≈ 58/60   (`raw_oracle_k3`)
  baseline ≈ 57/60

The 11/60 gap is the cost we pay for caching K/V across forward
passes. Three options:

1. **Accept the cost.** Cache amortisation (§5d) still pays off if
   per-doc query count is high — 18 % accuracy hit may be acceptable
   for ~7× faster prefill per query in long-tail RAG.
2. **Anchor-conditioned cache.** Build cache with the chunk plus a
   ~32-token surrounding window included in the forward; trim those
   when extracting cached K/V. This makes `h_a^cache` closer to a
   typical `h_b^assembled`. Untested.
3. **Hidden-state caching.** Cache `h_L` instead of K/V; recompute
   k_norm/k_proj/RoPE and v_proj at query time. Same drift problem
   in `h_L` but one indirection removed. Untested.
4. **Shift focus off the splice mechanism entirely.** §5h showed
   retrieval is the bigger lever for bookshop at top_k=6 (Jina's
   chunk_repr space mismatch). The chunk_repr ablation experiment
   is a separate axis from the splice quality work.

## 5m. Anchor-conditioned cache — closes 7/11 of the gap on oracle

§5j said the cache cost comes from `h_a^cache` ≠ `h_b^assembled`: the
cache forward sees the full-haystack context, the assembly forward
sees just sink + chunks + Q. To shrink that gap, change the cache
build: per chunk, run a small forward of `[first M tokens of doc] +
[chunk tokens]` instead of the one big haystack forward. Capture K/V
for the chunk positions only. Post-shift K by `(a - M)` so on-disk
format is identical to standard cache and the splice mechanism works
unchanged.

Edge case: chunk 0 starts at `a=0`, so prepending the sink would
duplicate the first M tokens. Special-case chunk 0 to a standalone
forward; its first M positions then *are* the sink K/V with no
preceding context — exactly what `build_sink_placement` needs.

Implementation: `build_anchor_chunk_cache` in `chunk_cache.py`,
exposed via `scripts/12_sink_mk.py --cache_kind anchor`.

### Direct K divergence (anchor cache vs current model)

Re-running the §5j diagnostic with anchor cache, mean cos(K_fresh, K_shifted):

| Layer | sink std | sink anchor | gold std | gold anchor | sib0 std | sib0 anchor |
|---|---|---|---|---|---|---|
| 3  | 1.000 | 1.000 | 0.886 | **1.0000** | 0.970 | 0.965 |
| 7  | 1.000 | 1.000 | 0.740 | **1.0000** | 0.888 | 0.934 |
| 11 | 1.000 | 1.000 | 0.512 | **1.0000** | 0.745 | 0.872 |
| 15 | 1.000 | 1.000 | 0.558 | **1.0000** | 0.751 | 0.882 |
| 19 | 1.000 | 1.000 | 0.670 | **1.0000** | 0.811 | 0.922 |
| 23 | 1.000 | 1.000 | 0.662 | **1.0000** | 0.806 | 0.925 |

Gold is now **numerically exact** across all 6 layers — anchor cache
fully removes the drift for chunks placed first after sink. Sib0
(placed second) and sib1 (placed third) still drift but less than
under standard cache.

### Accuracy

| Setup | standard | anchor v2 | Δ |
|---|---|---|---|
| sink_oracle_k3 (gold-first ordering) | 47/60 | **54/60** | **+7** |
| sink_k6 (Jina top-6, arbitrary order) | 43/60 | 35/60 | **−8** |

The oracle pipeline closes 7 of the 11/60 cache cost; remaining 3 are
from sib0/sib1 drift (chunks at positions 2/3 after sink) and a
couple of bookshop binding errors not related to the cache.

The retrieval pipeline regresses by 8. Anchor encodes "I am the first
chunk after sink"; in sink_oracle_k3 that's true for gold and the
result is great. In sink_k6, Jina orders by similarity — the gold
might land at position 2, 3, or 4 — so anchor mismatches the actual
placement. Across all 6 retrieved chunks, most aren't at position 1,
so the cache is wrong for most of them.

### What this opens up / closes

- **The cache cost is not structural after all.** §5l said the
  cache+splice ceiling was 47/60. With anchor it's at least 54/60 on
  oracle, possibly higher with better anchor designs. The 11/60 cost
  in §5j is largely a *build-time choice* (full-haystack forward),
  not a fundamental limit of cache-and-splice.
- **Anchor cache is placement-specific.** A cache built under
  assumption X about assembly positions will fail when X doesn't
  hold. The natural next move is multi-anchor cache (cache each
  chunk under several assembly positions, pick the matching one at
  query time) or a position-invariant cache build (e.g., anchor under
  a "typical" mixed-position context).
- **Reordering retrieval to gold-first would salvage anchor for
  sink_k6.** If we could put the gold chunk first in the assembly,
  sink_k6 would inherit sink_oracle_k3's +7. This couples cache
  design to retrieval ordering — a deliberate co-design rather than
  the orthogonal axes we have now.

## 5n. Anchor shape sweep — neither multi-anchor nor bridge buy much

§5m closed gold drift with a fixed [sink + chunk] anchor and reached
54/60 on oracle. Two follow-ups asked whether a smarter anchor shape
could move the needle further on the remaining 4/60.

**(a) Multi-anchor cheap probe (`filler_mode="self_prev"`).** Instead of
caching each chunk under "I am at position 1 after sink," cache it under
"I am at position 2 after sink with a plausible chunk in front of me":
`small_ids = [sink + chunk_{i-1} + chunk_i]`, K/V captured for the
chunk_i slice only, then RoPE-shifted to wherever it lands at query
time. Chunk 0 is special-cased (no sink), chunk 1 uses
`[chunk_0 + chunk_1]` (chunk_0 already begins with the sink). Exposed
via `--cache_kind anchor_prev` in `scripts/12_sink_mk.py`.

**(b) Bridge tokens at splice time.** Insert a fresh copy of the first M
doc tokens *between* every pair of cached chunks at splice time (not at
cache build). Each cached chunk thus sees an "anchor right before it"
in the assembled prompt, hopefully matching what the cache was built
against. Modeled via `build_chunk_placements_bridge` in
`scripts/12_sink_mk.py` with a new `bridge_oracle_k3` mode.

Both runs, full MK suite, anchor cache:

| Mode | total | vault | secret | bookshop |
|---|---|---|---|---|
| sink_oracle_k3 (anchor v2, §5m) | 54/60 | 19/22 | 10/10 | 25/28 |
| **anchor_prev_oracle_k3** (multi-anchor probe) | 54/60 | 19/22 | 10/10 | 25/28 |
| **bridge_oracle_k3** (M=4 fresh between chunks) | 53/60 | 18/22 | 10/10 | 25/28 |
| raw_oracle_k3 (no splice) | 58/60 | 22/22 | 10/10 | 26/28 |

Three variants of anchor shape all land in the 53–54/60 band. Whatever
gates the residual 4/60 isn't *which* anchor we build/insert — it's
something else inside the splice.

## 5o. V diagnostic + gold-only splice — siblings are the entire residual gap

Two final probes, both on `sink_oracle_k3` with anchor v2 cache.

**(a) V-side direct diagnosis.** §5j only measured K(fresh) vs
K(cache-shifted). Extended `scripts/13_diagnose_splice.py` to capture
`V_fresh` and `V_cached` at the same chunk positions. For gold (placed
first after sink) V cos = **1.0000** across all 6 layers — anchor cache
makes V exact, same as K. For sibling chunks (placed 2nd / 3rd after
sink) V cos ≈ 0.80, comparable to K cos. So V mismatch tracks K
mismatch one-for-one; it is not a separate, larger axis of drift.

Diagnostic data: `data/diag/splice_div_anchor_v2.json`.

**(b) Gold-only splice.** If sibling K and V are both drifted (cos ≈ 0.8)
and gold is exact, splice only the gold chunk's K/V; let the sibling
chunks be present as raw token ids that the model attends to fresh
during prefill. Same sequence length, same retrieval set, same
positions — just stop overwriting K/V at sibling positions. Implemented
as `gold_only_oracle_k3` mode in `scripts/12_sink_mk.py`.

| Mode | total | vault | secret | bookshop |
|---|---|---|---|---|
| sink_oracle_k3 (anchor v2, splice all 3) | 54/60 | 19/22 | 10/10 | 25/28 |
| **gold_only_oracle_k3** (splice gold only) | **58/60** | 22/22 | 10/10 | 26/28 |
| raw_oracle_k3 (no splice) | 58/60 | 22/22 | 10/10 | 26/28 |

Gold-only ties raw — splicing the gold chunk's K/V via anchor cache
has **zero accuracy cost**. The entire 4/60 residual after §5m comes
from splicing distractor siblings whose K/V drift to cos ≈ 0.8.

### Why distractor splice hurts more than expected

The drifted-K hypothesis would have predicted symmetric damage — gold
should also lose accuracy in proportion to its drift. With anchor v2
gold drift is gone (cos=1.0) and gold splice is free. But sibling drift
of cos ≈ 0.8 *is not equivalent to having a moderately-similar K
naturally*. Anchor-cached K/V at cos ≈ 0.8 sit off the model's training
manifold: the K vector still attends to the sibling chunk, but with
unnatural amplitude and direction. The model treats this as an
adversarial-style anomaly rather than as a normal distractor, which is
why pushing K/V through the *raw* forward (cos = 1 by construction) at
those positions recovers the lost cases.

Put differently: splice solves the "looking-at-me" problem (a chunk
needs its own K/V at the right query positions), not the
"looking-at-others" problem (a chunk needs to be looked at by neighbours
with realistic K/V). Distractors don't need to "look at themselves"
because they aren't the answer — they only need to be lookable-at by
the model. Spliced K/V degrades that lookability without compensation.

### What this changes about the path forward

- **Cache+splice ceiling moves from 47/60 to ≥58/60 on oracle.** §5l's
  pessimistic ceiling (47/60) and §5m's improved ceiling (54/60) both
  assumed "splice every cached chunk we have." Lift that assumption and
  the ceiling matches raw.
- **Per-chunk splice policy.** In real RAG (no oracle), the practical
  rule is: only splice chunks the system is confident about; let the
  rest pass through as fresh tokens. Sequence length and per-query
  compute are unchanged from full-splice, only the choice of *which
  positions to overwrite K/V at* changes.
- **Anchor cache is still load-bearing.** The 58/60 result only holds
  because *gold* is built under anchor v2 (cos = 1.0). Standard-cache
  gold at cos = 0.5 would still take a hit even if siblings are left
  alone. Anchor cache is what makes gold splice free; gold-only is
  what removes the sibling penalty.
- **Open follow-ups (not yet done):**
  - `gold_only` style applied to **sink_k6 (Jina retrieval)**: pick
    only the top-1 (or only the chunks whose Jina score crosses a
    threshold) for splice, let the rest pass fresh. If Jina's score
    ranking correlates with "is gold," this should beat
    sink_k6 (= 35/60 under anchor v2, regressed because most positions
    don't match anchor) without giving up cache amortisation.
  - "Why are off-manifold sibling K/V worse than no splice?" remains
    informally answered. A scaling probe (interpolate
    `K = α K_cached + (1-α) K_fresh` and measure accuracy vs α) would
    show whether the relationship is monotone or has a cliff.

Diagnostic data: `data/mk/gold_only_sink_oracle.json`,
`data/mk/anchor_prev_sink_oracle.json`, `data/mk/bridge_sink_oracle.json`,
`data/diag/splice_div_anchor_v2.json`.

## 5p. Random-anchor cache — drift correlation is not causal

§5o's 4/60 residual was attributed to "siblings' off-manifold K/V acts as
adversarial noise." A drift-direction probe on the existing anchor v2
diagnostic data (`data/diag/splice_div_anchor_v2.json` + new script
`14_drift_correlation.py`) sharpened the picture:

| layer | ‖dK_gold‖ | ‖dK_sib0‖ | ‖dK_sib1‖ | cos(sib0,sib1) | cos(gold,sib*) |
|---|---|---|---|---|---|
| L3 | 0.004 | 5.6 | 7.5 | **0.86** | ≈0 |
| L11 | 0.009 | 9.7 | 13.6 | **0.79** | ≈0 |
| L23 | 0.006 | 6.3 | 8.6 | **0.84** | ≈0 |

Sibling drifts share a strong common direction (per-head cos ≈ 0.80 in
head space). Hypothesis: this shared "I am a chunk right after sink, in
a fresh short forward" signature *is* what makes 3-chunk splice fail —
the model reads it as one unified off-manifold cue, not as 3 independent
distractors.

Test: per-chunk **unique random-token anchor** (`build_random_anchor_chunk_cache`
in `chunk_cache.py`; `--cache_kind anchor_random` and
`rnd_anchor_oracle_k3` mode in `scripts/12_sink_mk.py`). For chunk_i,
sample M random vocab tokens seeded by `(case_id, chunk_id)`, cache via
`[random_i + chunk_i]` short forward, splice with `random_i` inserted
fresh right before chunk_i in the assembly. If the shared-direction
hypothesis were right, decorrelating per-chunk drift directions should
recover accuracy.

**Result, full MK suite (60 queries):**

| Mode | total | vault | secret | bookshop |
|---|---|---|---|---|
| sink_oracle_k3 (anchor v2) | 55/60 | 21/22 | 10/10 | 24/28 |
| **rnd_anchor_oracle_k3** | **51/60** | 21/22 | 10/10 | **20/28** |
| raw_oracle_k3 | 58/60 | 22/22 | 10/10 | 26/28 |

Random anchor **regressed by 4/60** on bookshop. Drift re-measurement on
the random-anchor cache (`scripts/15_drift_correlation_rnd.py` →
`data/diag/drift_corr_anchor_random.json`):

| layer | ‖dK_gold‖ | ‖dK_sib0‖ | ‖dK_sib1‖ | cos(sib0,sib1) |
|---|---|---|---|---|
| L3 | 1.7 | 5.1 | 6.8 | 0.83 |
| L11 | 3.2 | 10.8 | 14.3 | 0.78 |
| L23 | 1.9 | 6.8 | 8.7 | 0.81 |

Two things drop out:

1. **The sibling-drift correlation stayed at ≈0.80** despite per-chunk
   unique anchors. Whatever drives the shared direction is not the anchor
   identity — it's a deeper geometric property of short-forward-built K
   at deep layers (likely "short-context K geometry" regardless of what
   the short context is). Decorrelating the anchor doesn't decorrelate
   the drift.
2. **Gold drift jumped from ≈0 to 1.7–3.2** because random anchor breaks
   gold's perfect-match property. In v2, gold's assembly prefix is
   `[global_sink]` which matches the cache build's `[sink_tokens]`. In
   random, gold's assembly prefix is `[global_sink + random_gold]` but
   cache build saw only `[random_gold]` — so h diverges at gold. We gave
   up gold's zero-drift for a sibling property that doesn't actually
   exist.

**Hypothesis disconfirmed.** Sibling drift direction shares a robust
geometry that anchor choice can't move. The accuracy cost of multi-chunk
splice is not driven by directional correlation.

## 5q. Anchor M sweep — drift magnitude is not causal either

If direction isn't causal, maybe **magnitude** is. Sweep
`--anchor_M ∈ {4, 8, 16, 32}` (the cache-build prefix length) on
sink_oracle_k3 with --M=4 sink at assembly held fixed.

| anchor_M | sink_oracle_k3 | bookshop | drift on M=32 |
|---|---|---|---|
| 4 (§5m baseline) | 54-55/60 | 24/28 | sib0: 5.6–10.1 |
| 8 | 54/60 | 24/28 | — |
| 16 | 54/60 | 23/28 | — |
| 32 | 54/60 | 23/28 | sib0: **4.5–8.2** (−15%) |

`scripts/14_drift_correlation.py` (with new `--cache_dir_prefix` flag) on
the M=32 cache (`data/diag/drift_corr_anchor_M32.json`):

| layer | ‖dK_gold‖ | ‖dK_sib0‖ | ‖dK_sib1‖ | cos(sib0,sib1) |
|---|---|---|---|---|
| L3 | 1.3 | 4.5 | 6.4 | 0.85 |
| L11 | 3.6 | 8.1 | 12.0 | 0.77 |
| L23 | 2.4 | 5.1 | 7.5 | 0.81 |

**Sibling magnitude did shrink ~15%** (sib0 5.6–10.1 → 4.5–8.2), but
**accuracy didn't move**. As a bonus null, gold drift went UP (0 → 1.3–
3.9) and accuracy STILL didn't move. So neither "smaller sibling drift
helps" nor "smaller gold drift helps" is causal at this magnitude scale.
Together with §5p:

- §5p: anchor identity → drift direction unchanged → accuracy WORSE
  (different reason: gold drift broke + gibberish tokens)
- §5q: cache-build context length → drift magnitude −15%, gold magnitude
  +large → accuracy unchanged

The 4/60 sibling-splice cost is robust across "how we cache" tweaks.

## 5r. α-blend cliff — and the actual decomposition of where speedup vs accuracy comes from

What if the binary "splice all or splice none" framing is the right one?
Tested by blending cached and fresh K/V at splice time:
`K_spliced = α · K_cached + (1−α) · K_fresh`, same for V. Added `alpha`
parameter to `assemble.patched_full_attn` and a `--reuse_cache` flag to
`scripts/12_sink_mk.py` so we can sweep α without rebuilding the cache.

| α | sink_oracle_k3 | per-q |
|---|---|---|
| 1.0 (full splice) | 54/60 | 1.12s |
| 0.75 | **58/60** | 1.15s |
| 0.5 | 58/60 | 1.16s |
| 0.25 | 58/60 | 1.17s |
| 0.0 (= raw, no splice) | 58/60 | 1.20s |

**Complete step function.** Any α < 1.0 immediately recovers raw
accuracy. The 4/60 cost lives entirely at α=1.0.

Mechanism. `Q · K_spliced = α · Q·K_cached + (1−α) · Q·K_fresh`. With
K_cached at cos≈0.8 vs K_fresh, the cached vector is 80% aligned but
20% off-axis. At α=1.0, Q has only the off-axis version to attend with —
20% misalignment is enough to scatter attention to the wrong sibling.
At α=0.75, the 25% `Q·K_fresh` term carries the correct
assembly-context signal and dominates the disambiguation; the
75% cached component is treated as direction-noise and tolerated. The
model needs *any* fresh signal at spliced positions; once present, the
cached admixture doesn't hurt.

### What this reveals about the value decomposition

The user's question — *if we use the splice format but skip K/V cache,
does the model still answer correctly?* — surfaces a sharper finding.

| Mode | tokens | per-q | acc |
|---|---|---|---|
| baseline (full prompt) | 8147 | 2.81s | 57/60 |
| raw_oracle_k3 (splice **format**, no K/V cache) | ~770 | 1.40s | **58/60** |
| sink_oracle_k3 α=0.5 (format + cache K/V) | ~770 | 1.16s | 58/60 |
| sink_oracle_k3 α=1.0 (format + full cache K/V) | ~770 | 1.12s | 54/60 |

Decomposing the contributions:

| Component | Speedup | Δ accuracy |
|---|---|---|
| Short assembly (vs full doc) | **2× (2.81 → 1.40s)** | +1/60 |
| Sink prepend | bundled above | (rescues degenerate decode, §5i) |
| Cache K/V replacement at α=1.0 | 1.25× (1.40 → 1.12s) | **−4/60** |
| α=0.5 blend (keeps cache, fixes accuracy) | 1.21× (1.40 → 1.16s) | 0 |

**Most of the win is from the FORMAT change**, not from the K/V cache.
The K/V cache replacement adds at most 17% per-query speedup AND, at
α=1.0, costs 4/60 accuracy. At α<1.0 the cache is essentially free on
both axes but its contribution is the marginal one.

This is a partially-negative result for the original ReAttention
premise: "cache K/V at original positions, splice in" is small relative
to "shorten the sequence and prepend a sink." The sequence-shortening
trick is essentially classic RAG (retrieve and stuff).

Side-by-side demonstration (case 0 / q0 — bookshop in Lisbon):
- baseline: 8147 tokens → 4.90s, correctly outputs "Linden Street"
- splice α=0.5: 788 tokens → 1.43s, correctly outputs "Linden Street"
- 10.3× shorter input, 3.4× faster, equal accuracy

### What's left for the cache K/V idea

Where K/V replacement might earn its keep (untested):

- **Longer chunks** (512, 1024 tokens) — chunk's standalone-text context
  diverges more from the original-doc context; cache's "restore original
  context" might matter more. See §5s.
- **Longer assembly contexts** (64K, 128K) — same logic, scaled up.
- **Cross-chunk reasoning tasks** — needles that require linking
  information across chunks. The raw chunk text loses the cross-references
  that cache-from-original might preserve.
- **Tighter retrieval noise regimes** — when retrieved chunks are not
  oracle and partial information is present.

The 8K + 256-tok + needle-style MK suite does not exercise these regimes.
The current verdict stands for this benchmark: cache K/V is edge
optimization, format is the win.

## 5s. Long-chunk benchmark — cache K/V loses what little edge it had

§5r left one open hypothesis: cache K/V might earn its keep on longer
chunks where each chunk loses more original-doc context. Tested at
chunk_size=512 (vs the standard 256) on the same MK suite_8k, same
α-blend sweep:

| chunk_size | α | sink_oracle_k3 | bookshop | per-q |
|---|---|---|---|---|
| 256 | 1.0 | 54/60 | 24/28 | 1.12s |
| 256 | 0.5 | 58/60 | 26/28 | 1.16s |
| 256 | 0.0 (= raw) | 58/60 | 26/28 | 1.20s |
| 512 | 1.0 | **52/60** | **21/28** | 1.23s |
| 512 | 0.5 | 58/60 | 26/28 | 1.30s |
| 512 | 0.0 | 58/60 | 26/28 | 1.31s |

Two things, both opposite to the "long chunk helps cache" prediction:

1. **The α=1.0 cost widened from −4/60 to −6/60**. Bigger chunk → larger
   slice of positions where K_cached is off-axis relative to assembled
   K_fresh → more attention misrouted. Cache K/V doesn't get "more
   right" as chunks grow; it gets more wrong.
2. **The per-query speed gap between α<1.0 (uses cache) and raw
   (doesn't) collapsed**. At chunk_size=256, cache provided 17% extra
   speedup (1.16 vs 1.20s). At chunk_size=512, the gap is 0.7% (1.30 vs
   1.31s — noise). The k_proj/v_proj cost the cache saves is fixed per
   position; per-query baseline grows linearly with assembly length, so
   the relative cache benefit shrinks as chunks get bigger.

### Combined verdict on the K/V cache mechanism

Three lines of evidence land on the same conclusion:

- §5p–q: drift direction and drift magnitude — robust to anchor tweaks,
  so we can't engineer the cache to be "more correct" via cache-build
  changes.
- §5r: α-blend cliff — the K/V replacement at α=1.0 specifically costs
  accuracy because it removes any assembly-context K_fresh component; at
  α<1.0 the cache contribution is essentially free but marginal.
- §5s: longer chunks make the α=1.0 cost worse AND the α<1.0 speed
  benefit disappear.

**On the 8K MK / 256–512-token-chunk / needle-style benchmark, the K/V
cache substitution is not the source of value.** What matters:

1. **Short-assembly format** (sink + retrieved chunks + Q ≈ 1–1.5K vs
   8K full doc) — 2× per-query speedup, +1/60 accuracy. This is what
   classic RAG already does.
2. **Sink prepend** — prevents degenerate decode (§5i).
3. **α<1.0 on the splice** — keeps the cache from being a footgun if
   we do use the cache.

The cache K/V replacement at any α is at best marginal speedup and at
α=1.0 a known accuracy cost. **Default `alpha=0.5` (or `alpha=0.0` =
just use raw retrieved chunks) is the right operating point** unless a
future benchmark shows cache K/V provides accuracy on harder regimes.

### Where cache K/V might still earn its keep (still untested)

- **Contexts ≥ 32K** with longer assembly (top_k ≥ 10) where the
  chunk-vs-original-doc context gap is much bigger, AND attention has
  enough positions to need the "correct original-doc K" hint.
- **Cross-chunk / multi-hop reasoning** — chunk-as-text loses links that
  cache-from-original might preserve.
- **Tight retrieval noise regimes** — when retrieved chunks are
  partial / wrong / overlapping.

None of these are testable on the current MK suite. Open question
whether they're worth building new benchmarks for or whether the
sprag project should re-scope around the format-change result and the
amortization curve (which is real and well-documented in §5d).

## 5t. 32K MK benchmark — format wins big, cache K/V's first non-negative signal

Generated a 32K MK suite (10 cases × 6 needles, ~32.7K tokens/case via
`gen_mk_suite --target_tokens 32768`). Ran baseline + sink_oracle_k3 at
α∈{1.0, 0.5} + raw_oracle_k3, anchor v2 cache, chunk_size=256, M=4.

| Mode | total | vault | secret | bookshop | per-q |
|---|---|---|---|---|---|
| baseline (full 32K) | 52/59 | 21/21 | 10/10 | 21/28 | 12.20s |
| sink_oracle_k3 α=1.0 | 52/59 | 19/21 | 10/10 | 23/28 | 1.11s |
| **raw_oracle_k3** | **56/59** | 19/21 | 10/10 | 27/28 | 1.12s |
| **sink_oracle_k3 α=0.5** | **57/59** | 20/21 | 10/10 | 27/28 | **1.08s** |

(1/60 queries skipped — gold chunk straddled a 256-tok boundary on
case 6.)

Four things drop out:

1. **Baseline accuracy genuinely degrades at 32K.** 57/60 (95 %) at 8K
   → 52/59 (88 %) at 32K. Bookshop is the casualty: 25/28 → 21/28. The
   model's attention over 32K of haystack scatters in the presence of
   multiple distracting needles. This is the regime sprag was always
   supposed to be useful in, finally exhibited.

2. **Short assembly wins on accuracy AND speed at 32K.** raw_oracle_k3
   (which is just classic RAG: oracle-retrieve 3 chunks, stuff into a
   short prompt, no cache K/V) gets **56/59 vs baseline 52/59 — +4
   accuracy** while running **11× faster** (1.12s vs 12.2s). Splice
   α=0.5 pushes to 57/59. Bookshop reaches 27/28 — near baseline-at-8K
   level — under both short-assembly modes.

3. **α=0.5 marginally outperforms raw at 32K (first non-negative
   signal for cache K/V).** raw=56, α=0.5=57. One case is noise-tier
   but the direction matches the §5r prediction that cache K/V might
   help once chunks lose more original-doc context. Splice α=0.5 is
   also a hair faster than raw (1.08 vs 1.12s). Worth retesting at
   chunk_size=512 / 64K to see if the +1 grows or stays at noise.

4. **α=1.0 footgun is ~−4/59 vs raw, same as 8K.** Long context didn't
   make the splice replacement either worse or better as a footgun —
   the cost scales with "how many spliced positions are off-axis,"
   which depends on chunk count and chunk size more than haystack
   length.

### Updated narrative for sprag

| Context | Baseline | raw_oracle | α=0.5 splice | Speedup vs baseline |
|---|---|---|---|---|
| 8K | 57/60 (95%) | 58/60 (97%) | 58/60 (97%) | 2.5× |
| 32K | 52/59 (88%) | 56/59 (95%) | 57/59 (97%) | **11×** |

At 8K the value-prop was thin (+1 accuracy, 2.5× speed). At 32K it's
substantial (+5 accuracy, 11× speed). Bookshop alone moves
21/28 → 27/28 — a 27 % relative improvement.

Notably the heavy lifting is still the format change (short assembly
+ sink), with cache K/V at α=0.5 contributing a marginal +1 accuracy
and ~4 % speedup. The §5r decomposition holds, just the magnitudes
are bigger. Sprag's value-prop, properly stated:

> **Multi-key long-context RAG over Qwen3.5-0.8B: short-assembly
> prompting (sink + retrieved chunks) wins +5/59 accuracy AND 11×
> speedup vs full-prompt baseline at 32K. Cache K/V at α=0.5 adds
> marginal accuracy and speed on top. Avoid α=1.0 (silent −4–6/59
> accuracy cost regardless of context length).**

## 5u. RGB benchmark — cache K/V is net-negative on real passages

First eval on an external RAG benchmark (RGB, chen700564, `data/benchmarks/rgb/data/en.json`,
300 records). Each record = a query + gold `answer` slots + `positive`
(gold) and `negative` (distractor) passages. We concatenate the shuffled
positive+negative into one ~8K-token noisy doc (median 40 passages),
build a standard full-doc chunk cache (chunk_size=256), and compare four
ways of answering with the **same** Jina top-5 retrieval where applicable.
Scoring = RGB checkanswer (every answer slot's alias present in output).

Code: `src/sprag/rgb.py` (loader+scorer), `scripts/16_rgb_eval.py`.

```
mode             acc          per-q   avg_tok
baseline         259/300 86.3%  3.87s   8213    full noisy doc
raw_topk         235/300 78.3%  1.95s   1291    sink + top-5, FRESH K/V
splice_topk      226/300 75.3%  1.94s   1291    cached K/V, α=0.5
splice_topk_a1   220/300 73.3%  1.94s   1291    cached K/V, α=1.0
```

Decomposition (each step uses the same chunks as the one above it):

- **baseline → raw_topk: −8 pts.** Pure retrieval recall — Jina top-5 over
  ~32 chunks misses gold passages (RGB has up to 34 golds + heavy noise +
  12 multi-slot integration questions). Orthogonal to the splice; it's the
  cost of doing RAG at all vs. stuffing everything. 2× speedup.
- **raw_topk → splice_topk (α=0.5): −3 pts.** The clean cache-K/V cost,
  identical retrieved chunks. On *real* passages the cache K/V is
  **net-negative even at α=0.5** — the opposite of the MK suite (§5r,
  where α<1.0 was free). The MK "cache is free" result leaned on oracle
  retrieval + synthetic single-needle chunks; it does not transfer.
- **α=0.5 → α=1.0: −2 more.** The §5r footgun **replicates** on a real
  benchmark. α=1.0 remains strictly dominated.

**Takeaway:** the §5t/§5r story ("format is the win, cache K/V is edge")
holds *directionally*, but on real RAG the cache K/V edge flips negative.
Sprag's defensible value-prop is the **short-assembly format + sink**
(2× speed, recall-bounded accuracy); the K/V splice is a speed micro-opt
that costs a few points of accuracy and should be presented as such, not
as the mechanism. [[sprag-splice-decomp]]

## 5v. Per-subspace α — the RoPE split is not a usable lever

Probe (idea: blend only the rotated 64 dims or only the 192 pass-through
dims of K). Qwen3.5 here: head_dim=256, partial_rotary_factor=0.25 →
rotary dims [0:64) rotated, [64:256) pass-through; RoPE touches **Q,K
only, never V**. After `shift_rope` both `k_cached` and `k_fresh` sit at
the same phase R_b, so a post-RoPE blend is a linear interp of the raw-K
*content* split along the rotary boundary (`assemble.py` `alpha_k_rot` /
`alpha_k_pass` / `alpha_v`; `scripts/12` flags). All configs below share
the same freshly-built standard 8K caches (sink_oracle_k3, M=4 S=4), so
within-sweep comparison is valid.

```
cfg  K-rot   K-pass   V       correct/60
A    cached  cached   cached      47      footgun ref (all cached)
E    0.5     0.5      0.5         50      blend everything (recovery ref)
B    cached  fresh    cached      35      freshen pass-through K only
C    fresh   cached   cached      28      freshen rotary K only
D    fresh   fresh    cached       5      freshen ALL K, V cached
P    0.5     cached   0.5         43      blend rotary K + V
Q    cached  0.5      0.5         45      blend pass-through K + V
```

Two stacked coherence constraints kill the subspace lever:

1. **K–V coherence** (B/C/D): freshening K while V stays cached is
   catastrophic — D = 5/60. Fresh K comes from the short-assembly
   hidden-state regime, cached V from the full-doc regime; attention
   weights from one regime indexing value vectors from the other is
   garbage. (Matches the prior committed finding "K-V coherence is
   non-negotiable".)
2. **Within-K cross-dimension coherence** (P/Q): even with V riding along
   (blended 0.5) to preserve K–V coherence, blending *one* K subspace
   while caching the other lands P=43, Q=45 — **below** the all-cached
   A=47, not between A and E. Splitting K's blend at the rotary boundary
   introduces an internal-K incoherence that costs more than it buys.

**Conclusion:** the α-blend is an all-or-nothing *direction* knob (how far
the whole cached K/V moves toward fresh), **not** decomposable by RoPE
subspace. Q (45) > P (43) is the only within-subspace signal — weakly
hints rotary-K is the more splice-tolerant half — but it's 2/60 on a
shallow cliff (47→50) and not worth leaning on. Caveat: this 8K
standard-cache cliff (47→50, +3) is shallower than §5r's anchor-cache
cliff (54→58, +4); absolute numbers differ by cache type, the ordering is
internally consistent.

> **Correction (see §5w):** the "K and V are an indivisible pair" reading
> below in the K-vs-V decomposition is a *standard-cache* artifact. On the
> anchor cache, V can be cached independently (v_only = 58/60); the residual
> splice cost is K-only. The real axis is cache→assembly *drift*, not a K–V
> binding law.

### K-vs-V decomposition (standard 8K cache)

Does the splice cost come from K or V? Split via `splice_kind` "k"/"v"
(`k_only_oracle_k3` / `v_only_oracle_k3`), α=1.0, same standard 8K caches:

```
config              K       V       correct/60   per-q
raw_oracle_k3       fresh   fresh       58        1.36s   upper bound
sink_oracle_k3      cached  cached      47        1.75s
k_only              cached  fresh       29        1.77s
v_only              fresh   cached       5        1.78s
```

On standard cache, **both single-splice configs are worse than splicing
both** (29, 5 < 47), and v_only (fresh K + cached V) is catastrophic. Read
naively this says "(K,V) is an indivisible pair, no reusable half." §5w
shows that reading is cache-specific. Note also: splice is *slower* than
raw here (1.75 vs 1.36s) — at 8K short assembly our impl recomputes fresh
K/V then overwrites, so the cache is pure overhead; its speedup only
materializes when prefill is actually skipped (32K amortization, §5d/§5t).

## 5w. The splice cost is cache→assembly DRIFT, not a K–V binding law

Re-ran the K-vs-V decomposition (§5v) on the **anchor** cache
(`cache_kind=anchor`, `[sink+chunk]` per-chunk forward) vs the **standard**
full-doc cache. Same 8K suite, sink_oracle_k3, M=4 S=4, α=1.0.

```
config       K       V       standard   anchor
raw          fresh   fresh      58         58
both cached  cached  cached     47         54
k_only       cached  fresh      29         54
v_only       fresh   cached      5         58   ⬅ flips
E (α=0.5)    0.5     0.5        50         58
```

On the anchor cache the story inverts: **accuracy is governed almost
entirely by K's freshness, and V is freely cacheable.**

```
K state            anchor acc
fresh (v_only/raw)    58
blended (E)           58
cached (k_only/both)  54
```

v_only (cached V + fresh K) = 58 on anchor vs **5** on standard. The §5r
4-point cliff is a **pure K phenomenon** here (cached K at cos≈0.8 misroutes
attention; any fresh-K admixture fixes it) — exactly the §5r mechanism note.
V was never the problem on anchor.

**The unifying variable is drift, not coherence.** The §5v/§5v-decomp
"indivisible pair" was a symptom of how far the cache sits from the
assembly context:

- **standard cache**: K/V built in full-doc context. cached V encodes "this
  token after the entire document"; pairing it with fresh (assembly-context)
  K is a severe mismatch → v_only collapses to 5. Both K and V drift hard,
  so breaking the pair is catastrophic.
- **anchor cache**: K/V built in `[sink+chunk]`, *near* the assembly
  context. anchor-V ≈ assembly-V, so cached V + fresh K is fine (58). Only a
  small residual K drift remains (54), erased by any blend.

This is the original ReAttention premise vindicated and sharpened: **build
chunk K/V in a near-deployment context (anchor-style) and the splice is
viable; the residual cost lives in K and is removed by α<1.0.** Standard
full-doc caching drifts too far and is net-harmful (§5u RGB confirms on real
passages). Corollary: on anchor cache you *can* cache V and recompute K
fresh (v_only, 58) — but that saves only v_proj, not the prefill, so it's
not an efficiency win; the scientific value is localizing the cost to K.

Open: does this replicate on RGB? §5u showed standard-cache splice is
net-negative on real passages; the §5w prediction is that **anchor-cache
splice should close most of that gap on RGB**. Long-running validation
queued (`scripts/16_rgb_eval.py --cache_kind anchor`, K-vs-V modes).
**Answered in §5x: it does NOT — the drift/coherence story is a
synthetic-suite artifact and does not transfer to real passages.**

## 5x. RGB validation — the §5w drift story does NOT generalize

Ran `scripts/16_rgb_eval.py --cache_kind anchor` over all 300 RGB en.json
records (same Jina top-5 retrieval), adding `k_only_topk` / `v_only_topk`,
to test the §5w prediction (anchor cache should rescue the splice). It is
**falsified**. Side-by-side with the §5u standard-cache numbers:

```
config        K       V       RGB-standard(§5u)   RGB-anchor
raw_topk      fresh   fresh        78.3%             78.3%
blend α=0.5   0.5     0.5          75.3%             75.3%
α=1.0         cached  cached       73.3%             69.3%   ⬅ anchor WORSE
k_only        cached  fresh          —               73.0%
v_only        fresh   cached         —               72.7%
```

All three MK phenomena collapse on real passages:

1. **Anchor does not rescue the splice.** α=0.5 is identical (75.3%) and
   α=1.0 is *worse* on anchor (69.3 vs 73.3) — opposite of MK (§5w: anchor
   54 > standard 47).
2. **K and V are symmetric** (k_only 73.0 ≈ v_only 72.7). The MK K-vs-V
   asymmetry is gone.
3. **No coherence catastrophe.** v_only = 72.7%, not the MK-standard
   5/60 collapse. Breaking the pair is harmless here.

RGB-anchor is cleanly monotonic: raw (78.3) > blend (75.3) > single-splice
(~73) > both-cached (69.3). **More fresh = strictly better.**

**Why the MK phenomena were artifacts.** The drift/coherence framing
assumed a single coherent document in which a chunk's K/V meaningfully
encodes its in-document context — which anchor caching can approximate, so
"build near the assembly context" helps (MK). RGB "documents" are
concatenations of *unrelated* passages; there is no original-doc context
to preserve, the ReAttention premise is vacuous, and any cache K/V is just
a staler version of fresh → monotonically harmful, K/V symmetric, no
coherence to break. The synthetic MK haystack manufactured the drift,
coherence, and K-vs-V structure; real RAG has none of it.

### Robust cross-benchmark conclusion

| benchmark | doc type | cache K/V splice |
|---|---|---|
| MK 8K/32K | one coherent synthetic doc | small +/− around format; anchor helps, α=1.0 footgun |
| RGB | concatenated unrelated passages | **monotonically harmful**, all cache variants |

Across MK and RGB the only consistently positive component is the
**short-assembly format + sink** (2× speed at 8K, 11× at 32K, recall-bounded
accuracy). The cache-K/V splice is at best a long-context speed micro-opt
(only when prefill is actually skipped, §5d/§5t) and on real passages is a
pure accuracy cost — no cache construction (standard/anchor), no
K/V/subspace decomposition rescues it (§5u/§5v/§5w/§5x). Sprag should be
framed as **format + sink**, with the K/V cache presented honestly as the
incremental/negative piece it is. [[sprag-splice-decomp]]

## 5y. Fixed symmetric anchor — build≡use only fixes the top-1 chunk

§5w's anchor cache used the doc's first M tokens as the per-chunk sink
(content-dependent, chunk 0 special-cased). Tested a cleaner, fully
symmetric variant — `cache_kind=anchor_fixed`, `build_fixed_anchor_chunk_cache`:
one FIXED content-independent token (`<|endoftext|>`×M — Qwen3.5 has no BOS,
so endoftext is the doc-boundary/sink analog) leads every chunk's build
forward (no chunk-0 special case), and the SAME fixed anchor is placed once,
FRESH, at the front of the assembly (`build_sink_assembly`). The anchor sits
at position 0 with nothing before it in both build and use, so its K/V is
bit-identical across the two → the top-1 chunk sees exactly its build context
(zero cache→assembly drift). suite_8k, oracle k=3, M=4:

| config        | K      | V      | standard | doc-sink anchor | fixed anchor |
|---------------|--------|--------|----------|-----------------|--------------|
| raw           | fresh  | fresh  | 58       | 58              | 59           |
| both (α=1.0)  | cached | cached | 47       | 54              | 56           |
| E (α=0.5)     | 0.5    | 0.5    | 50       | 58              | **60**       |
| k_only        | cached | fresh  | 29       | 54              | 52           |
| v_only        | fresh  | cached | 5        | 58              | 58           |

**Fixed ≈ doc-sink anchor within n=60 noise; the symmetric front anchor does
NOT close the K-splice gap.** at α=1.0 both-cached 56 / k_only 52 still sit
below raw 59 — the same residual K cost §5w localized. Reason: a single front
anchor makes build≡use ONLY for the top-1 chunk. In oracle-k3 (gold + 2
siblings) the chunks landing at assembly positions 2–3 were each built as
`[anchor + chunk_i]` (seeing only the anchor in front) but deployed behind 1–2
other chunks, so their K is still computed in the wrong local context — that
unaddressed multi-chunk drift is the surviving ~3/60. Making *all* chunks
symmetric would require building each in its real multi-chunk assembly context
(≈ `anchor_prev`, §5n), defeating the precompute-once win.

Two robust takeaways:
1. **The anchor pattern is content-independent.** V freely cacheable
   (v_only 58 = raw) and cost localized to K (k_only/both < raw) reproduce
   under a fixed token, so §5w's effect is about the per-chunk *near-context
   build*, not what the sink says. Standard full-doc caching (v_only 5) remains
   the only catastrophic construction.
2. **`E (α=0.5) = 60/60` is the best cell in the whole §5w/§5y grid** — but
   that's the α<1.0 blend doing the work (§5/§5t: admixing fresh K restores the
   assembly signal), not the fixed anchor per se; doc-sink E was already 58.
   Fixed anchor is a *cleaner* construction (content-independent, no chunk-0
   special case, anchor K/V precomputable once across all docs) but not a
   *better* one. As always the blend, not the cache, carries the accuracy.
   [[sprag-splice-decomp]]

### 5y-RGB. The MK "E(0.5) ≥ raw" win does NOT survive real passages

The one place fixed anchor looked like it might beat fresh — MK E(α=0.5)=60/60
vs raw 59 — was a synthetic-suite artifact (the §5x warning). Validated on all
300 RGB en.json records (`scripts/16_rgb_eval.py --cache_kind fixed`, same Jina
top-5, α=0.5):

| cache_kind | raw | E(α0.5) | α=1.0 | k_only | v_only |
|------------|-----|---------|-------|--------|--------|
| standard (§5x) | 78.3 | 75.3 | 73.3 | — | — |
| anchor (§5x)   | 78.3 | 75.3 | 69.3 | 73.0 | 72.7 |
| **fixed**      | 75.0 | 68.7 | 64.0 | 63.0 | 66.7 |

On RGB the fixed anchor is **monotonic-harmful AND uniformly worse than
standard/anchor at every cell**: raw 75.0 > E(0.5) 68.7 > v_only 66.7 >
both-cached 64.0 > k_only 63.0 — more fresh = strictly better, blend 6.3 pts
*below* raw (vs MK where it was +1). Two reasons fixed is *worse* than the
doc-sink anchor here:
1. **The `<|endoftext|>` front anchor is a worse sink than the doc's natural
   lead tokens even on the fresh path** — raw alone drops 78.3 → 75.0 (the only
   difference for raw_topk is the M front tokens; everything else, incl.
   retrieval reprs, is identical). A content-bearing sink primes better than a
   bare boundary token.
2. The `[<|endoftext|> + chunk]` build context is *further* from the RGB
   assembly context (unrelated neighbour passages) than `[doc-lead + chunk]`,
   so the cached K/V drifts more.

**Settles the "is it valuable?" question: no.** The fixed symmetric anchor's
sole apparent advantage was an MK artifact; on real RAG it underperforms the
plain doc-sink anchor *and* fresh. Confirms §5x — symmetric build/use only
de-drifts the top-1 chunk and only matters when there is coherent in-doc
context to preserve, which real concatenated-passage RAG lacks. Robust
conclusion unchanged: **format + sink (content-bearing) is the win; any cache
K/V splice is incremental-to-negative, and a bare-token anchor is negative.**
[[sprag-splice-decomp]]

## 5z. Blending the LINEAR-attn state — the other 18 layers, finally probed

Everything in §5–§5y splices only the **6 full-attn layers**; the **18
GatedDeltaNet (linear-attn) layers were always recomputed fresh**. §5z asks the
parallel question for them: cache each chunk's recurrent state and blend it into
assembly, `S_used = α·S_cached + (1−α)·S_fresh` per linear layer (the
apples-to-apples analog of the K-blend).

**Mechanism / why it's not a clean splice.** Unlike K/V — a per-position tensor
whose chunk slice can be RoPE-shifted into any assembly position — the
GatedDeltaNet state is a single *gated sequential fold* over the whole prefix
(`modeling_qwen3_5`: `S = S·g.exp() + k⊗v`). There is **no position-independent
per-chunk slice**; a chunk's only cacheable unit is its *from-zero* fold, and
using it forces a composition rule. Code: `assemble.compute_chunk_linear_states`
(from-zero fold per chunk) + `assemble.patched_linear_state` (blends the
end-of-prefill state the *decode* reads — leaving the context fold fresh, the
exact parallel to how `patched_full_attn` leaves context hidden states fresh but
overwrites the K/V Q attends to). `scripts/17_linear_blend.py`, suite_8k oracle
k=3, full attention left **fresh** so all movement is the linear state alone.
Composition = **sum** of the retrieved chunks' from-zero folds; `--norm_match`
rescales the composed state to the fresh state's per-head Frobenius norm.

| compose | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---------|-----|--------|-------|--------|-------|
| sum            | 58 | 54 | 43 | 25 | **4** |
| **norm-matched** | 58 | 57 | 54 | 53 | **17** |

(α=0 = 58/60 both, the exact fresh sanity baseline — `patched_linear_state` at
α=0 is a verified no-op.)

**Findings:**
1. **The naive `sum` composition is scale-broken.** Summing k from-zero folds
   gives a state ~k× over-scaled (and undecayed), which is most of the sum-row
   collapse: norm-matching alone lifts α=0.5 from 43→54 and α=0.75 from 25→53.
2. **Even norm-matched, blending is "free at best, never a win."** At α≤0.5
   it's within ~4 of fresh (57/54 vs 58) but always ≤ fresh — it rides along
   harmlessly, it does not help.
3. **Pure cached state (α=1.0) collapses regardless: 4/60 sum, 17/60 norm-
   matched.** So the composed *direction* is genuinely wrong — a sum of isolated
   from-zero folds cannot reconstruct the true sequential gated fold — and the
   fresh fold is the only thing carrying the answer. You **cannot skip the
   linear fold.**
4. **Strictly worse than the full-attn K-blend.** On the same MK setting the
   K-blend held ~flat to α≈0.75 and only mildly dropped at α=1 (anchor: 58→54);
   the linear blend erodes through the mid-range and collapses to 17 at α=1.
   Reason: K/V have positional locality (cached chunk K/V at RoPE-shifted
   positions ≈ fresh), the linear fold has none.

**Verdict.** Caching/blending the linear state offers **no accuracy upside**
(strictly ≤ fresh even on saturated MK) and **no compute upside** (the α-blend
needs the fresh fold, so nothing is skipped — same structural limitation as the
K-blend at α<1). The 18 linear layers are best left fresh. This *extends* the
project's robust conclusion to the whole architecture: across both attention
families, the cache splice is incremental-to-negative; **format + content-
bearing sink is the win.** Note MK is saturated here (§5-pt1: α=0 already
96.7%), so MK can only show "free, not better" — an RGB run would show the
expected net-harm in a discriminative range, but the value-prop is already dead
on the no-compute-skip ground. [[sprag-splice-decomp]]

## 5aa. full−pos+fresh delta cache — a residual that cancels position, in theory

User-proposed (2026-05-30). Build TWO caches per chunk and take their residual:
- **full** = chunk's rep in `[anchor][real preceding context][chunk]` (sees real ctx).
- **pos**  = chunk's rep in `[anchor]⟨position gap⟩[chunk]` — same absolute
  position (a position-id gap holds the slot), anchor kept, context tokens
  erased. Realized as: feed `[anchor]+[chunk]` as a sequence (so the chunk only
  attends to the anchor) but give the chunk its ORIGINAL position_ids.
- **use**: `full − pos + fresh`. `full − pos` subtracts the same-position,
  anchor-only baseline → the real context's *content contribution*; `+ fresh`
  re-bases onto the new assembly. Intent: keep the cross-context "memory," drop
  stale position, paste new position. Applied to all three targets (K/V residual
  ADDED onto fresh K/V — K residual `shift_rope`'d to the new position first;
  linear residual = Σ over retrieved chunks of `full_S − pos_S`, ADDED onto the
  fresh fold). `scripts/18_delta_cache.py` (`--target kv|linear|both`),
  `assemble.DeltaPlacement` / `patched_full_attn_delta` /
  `patched_linear_state(additive=True)` / `compute_running_linear_states`.
  α scales the residual; α=0 = pure fresh (verified exact sanity).

suite_8k oracle k=3, M=4, full attn otherwise fresh, n=60:

| α | K/V | linear | both |
|------|-----|--------|------|
| 0    | 59 | 59 | 59 |
| 0.25 | 51 | 57 | 48 |
| 0.5  | 36 | 52 | 34 |
| 0.75 | 18 | 40 | 11 |
| 1.0  | **6** | **15** | **3** |

**All three monotone-harmful; none beats fresh anywhere.** Linear is the
gentlest (tolerates the residual longest), K/V steeper, **both** steepest (the
two perturbations compound — same super-additive interaction as the stacked
splice in the §5z follow-up). The *literal* method (α=1) collapses every target
(K/V 6, linear 15, both 3 — degenerate repetition like "\nA:\nA:…").

**Why it fails — and it's NOT a position problem.** The K residual is
`shift_rope`'d to the new assembly position, so `full−pos` and `fresh` are both
at the correct new position; there's no rotation conflict. The failure is
**magnitude / double-counting**: the method assumes `fresh` is a context-poor,
"position-only" representation that the residual completes — but `fresh` is
already a COMPLETE rep (the chunk having attended to its *new* assembly
neighbours). Adding the old context's marginal contribution on top yields an
*over-complete* K/state (~2× content), which over-drives attention scale (K) or
the recurrent fold (linear) and breaks decoding. To be coherent the `+fresh`
term should instead be the **new-position anchor-only baseline** (`pos` rebuilt
at the assembly position), giving `pos_new + (full−pos) ≈ full rebased` — but
that is just the ordinary cached-chunk splice (§5w), which we already know is
incremental-to-negative. So the residual construction adds nothing the plain
splice doesn't, and the additive form is strictly worse.

**Empirical confirmation of the diagnosis** (`--mode replace`, K/V, n=60).
Replacing `+fresh` with `pos_new+(full−pos)=shift(full)` (i.e. REPLACE the fresh
K/V with the shifted real-context cache instead of ADDING the residual onto it):

| α | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|------|-----|------|-----|
| add (fresh + residual)     | 59 | 51 | 36 | 18 | **6** |
| replace (= shift(full))    | 59 | 53 | 44 | 32 | **29** |

Same residual information, but the coherent REPLACE form **degrades gracefully
into the standard-splice band (29/60 at α=1, ≈ §5w standard full-doc cache)
instead of collapsing (6/60)**. This isolates the cause: the collapse was the
additive double-counting against an already-complete `fresh`, not the residual
itself. And the punchline survives — even the graceful replace is still ≤ fresh
(29 < 59), i.e. the residual idea *made coherent is the standard splice*, which
is incremental-to-negative.

**Decomposing the 29 (here) vs 54 (anchor cache, §5w) gap** — replace α=1, n=60,
adding the §5w setup knobs one at a time:

| setup (replace, α=1) | correct/60 |
|----------------------|-----------|
| eot sink, no strip (here) | 29 |
| + strip 4 (drop chunk head) | 28 |
| + strip 4 **+ doc-lead sink** | **41** |
| §5w standard cache (for ref) | ~47 |
| §5w **anchor** cache (near-context) | **54** |

Surprise: **chunk-strip is ≈ inert here (29→28); the sink swap (eot→doc-lead)
does the work (→41)** — a content-bearing sink primes generation better (cf.
§5y). So 29→41 is pure *setup* (dominated by the sink). The residual **41→54
gap is the genuine drift**: this experiment's `full` cache is built in the
chunk's *entire real preceding document* (far from the short assembly), whereas
§5w's anchor cache is built in a *short near-deployment context* `[anchor][chunk]`
(≈ assembly) → low drift. The delta method is structurally married to the
high-drift `full` cache: if `full` used the low-drift anchor context, `full≈pos`
and the residual vanishes. So it can never reach the anchor cache's 54/60 (let
alone §5y fixed-anchor's E(0.5)=60, which also leaned on an α=0.5 *blend* + that
near-context build). Consistent with the whole arc: **every cache manipulation —
replace, blend, or residual-add, on K/V, linear, or both — is free-at-best and
never better than fresh recompute over the short assembly.** [[sprag-splice-decomp]]

**`pos` construction is inert (`--pos_fill anchor`).** Tried filling the erased
context slots with anchor placeholders (`[anchor][eot×a_start][chunk]`, natural
positions) so `pos` matches `full`'s key-count/position and differs only in
context *content* — vs the default position-id gap (chunk attends to only the M
anchor keys). add-mode K/V n=60: gap `[59,51,36,18,6]` vs anchor-fill
`[59,49,37,23,8]` — **identical within noise.** Two reasons: (1) the additive
over-count dominates and is insensitive to how `pos` erases context; (2) the
placeholders are all the *same* token, so a chunk attending to N identical `eot`
keys ≈ attending to the M `eot` anchors (softmax over identical keys) → `pos_anchor
≈ pos_gap`. Dead end either way: identical fill ≈ gap; *varied* fill would inject
its own content (no longer erasing). The gap already captured the clean baseline.

## 5d. Amortization sweep (16K, 8 queries / doc)

The headline value-prop test from §7.2. One 16,333-tok haystack with 8
distinct needles (2 booksh ops, 5 vault-numbers, 1 secret-keeper) at
evenly-spaced depths; 8 queries against the same doc.

Driver: `scripts/data/gen_amortization.py` + `scripts/06_amortization_sweep.py`.
Cache build is one-shot; per-query times are measured separately.

| Mode | Cache | Per-query | Total (8 q) | Accuracy |
|---|---|---|---|---|
| baseline | — | 5.50 s | 44.0 s | 6/8 |
| reattn top-3 | 16.2 s | 1.84 s | 30.9 s | 2/8 |
| reattn top-6 | 17.2 s | 1.99 s | 33.1 s | 5/8 |

Break-even ≈ 4–5 queries (cache / (per-q baseline − per-q reattn)).
At 8 queries ReAttention is already ~25–30 % faster end-to-end, and the
asymptote is 2.8× faster per query.

**Accuracy gap — diagnosis, not ReAttention's fault.** All 5
vault-number queries retrieved nearly the same chunk set
`{9, 16, 63, ...}` regardless of which vault was asked. Jina embeds
"What magic number is stored in vault X" the same way for every X — the
template dominates over the disambiguating token. The model then sees
multiple vault-needle chunks competing in context and hallucinates
across them ("Delta is forty-two" when Delta is actually 101).
Bumping top_k=3→6 includes the right needle more often (2/8 → 5/8) at
negligible per-query cost. Top-k saturation is the right knob here, not
a ReAttention change. Two of baseline's "correct" answers (q2, q5) are
also scoring artifacts — model says "101" but the gold string is
"one hundred and one"; both modes are penalised by the substring matcher
the same way.

Counts of baseline correct that ReAttn missed are concentrated on the
vault-template family. The amortization curve is robust; the retrieval
discrimination is the next bottleneck (also flagged in §7.5
chunk_repr ablation).

## 5c. Turing SDPA workaround
PyTorch's mem-efficient SDPA backend doesn't support
`enable_gqa=True` on Turing — transformers' fast path then falls
back to MATH which materialises an N×N score matrix and OOMs at
16K. `loader.load_model()` patches
`transformers.integrations.sdpa_attention.use_gqa_in_sdpa →
False` on cap < 8.0, forcing the path that explicitly
`repeat_kv`'s to 8 heads. Mem-eff then handles 16K in ~400 MB and
32K in ~1.5 GB. The patch is scoped per-process; short-seq
submodels (Jina embedder) still use the math backend.

## 5. File map

```
src/sprag/
  loader.py            Text-only Qwen3.5 load (skips vision tower)
  rope.py              Inverse-RoPE shift + partial-rotary handling
  embed.py             Jina-embeddings-v5 wrapper (retrieval task)
  chunk_cache.py       Per-chunk K/V extraction + safetensors
  retrieve.py          Cosine top-K
  assemble.py          Patched full-attn forward (Inverse RoPE splice)
  runner.py            End-to-end runner: retrieve → assemble → generate
  mags/calibrate.py    Capture residuals, SVD-fit B/μ/τ
  mags/intervene.py    Online forward_hook projection
scripts/
  00_sanity_forward.py
  01_build_chunk_cache.py
  02_smoke_runner.py        # Octavia/forty-two needle proof
  03_run_niah.py            # baseline / reattn / full
  04_calibrate_mags.py      # SVD over T+/T- on NIAH-like pairs
  05_inspect_mags_fire.py   # debug: distance + fire rate per layer
  data/gen_niah.py          # minimal NIAH generator (single-needle)
  rescore.py
tests/
  test_rope.py
  test_identity_assembly.py
data/
  niah/                # generated cases + result jsonl (small, committed)
  mags/mags_4k.pkl     # the 8-pair calibration (small, committed)
```

## 6. GPU porting checklist

1. **Device placement** — `load_model(...)` returns the model on CPU.
   Add `model.to("cuda")` or pass `device_map="cuda"` in
   `from_pretrained`. `JinaEmbedder(device="cuda")` already exists.
   All splice/RoPE tensors already do `.to(key_states.device)`.
2. **bf16 vs fp16** — the model is bf16 by default. On older GPUs use
   `dtype=torch.float16` in `load_model`.
3. **flash-linear-attention** — `chunk_gated_delta_rule` falls back to
   a torch loop on CPU. On GPU, install
   `flash-linear-attention` and `causal-conv1d` for the fast path
   (transformers warns at load time about which library is missing).
4. **Memory** — at 32K context the K/V cache for the 6 full-attn
   layers is ~6 MB per chunk; the linear-attn state is bigger.
   With bf16 and ~50 chunks the prefill peak is around 4 GB
   activations + 1.6 GB weights — comfortable on any 16 GB+ GPU.
5. **Attention impl** — switch from `attn_impl="eager"` to `"sdpa"`
   in `loader.py` for the GPU runs (the eager fallback is fine for
   patching, but SDPA is 2-3× faster).

## 7. Experiments worth running on GPU

1. **16K / 32K NIAH** — where baseline actually starts to fail and
   ReAttention's amortization matters. Use the same harness:
   `scripts/data/gen_niah.py --target_tokens 32768`.
2. **Amortization sweep** — one long doc (~32K), 50 different
   queries. ReAttention pays the chunk-cache build once; baseline
   pays full prefill 50 times. The asymptotic time ratio is the
   real value of the approach. *First 16K/8-query data point in §5d
   — break-even ~4–5 queries; 32K + larger N pending.*
3. **RULER multi-needle / multi-key** — single-needle is too easy.
   Multi-needle / multi-value forces cross-chunk integration, where
   semantic drift (and therefore MAGS) is more likely to matter.
   *First 10×6 data point in §5e — bookshop template fails 19/28 at
   top_k=6, mostly via degenerate decode.*
4. **Better MAGS calibration** — 50+ pairs, longer contexts. Tune
   `--k_svd ∈ {4, 8, 16}`, `--tau_quantile ∈ {0.90, 0.95, 0.99}`,
   and apply `alpha ∈ {0.5, 1.0, 1.5}` at intervention.
5. **chunk_repr ablation** — currently uses jina embeddings.
   Already store `repr_mean_last` (Qwen's own last-layer mean-pool).
   Compare retrieval accuracy with both.
6. **LegoLink v2** — implement (G_c, M_c) decomposition cache for
   the 18 linear-attn layers. Measure whether the chunk-state
   reuse changes accuracy vs the v1 re-forward.

## 8. Known limitations / open questions

- **Inverse RoPE is provably exact on rotated dims**, but we have not
  empirically validated that "Inverse-RoPE shifted K" is the
  *semantically correct* K for the new position when the chunk's
  hidden state would differ under the new context. The design
  argument is that we *want* the original-context K (this preserves
  the chunk's semantic representation), but ablation is needed.
- **chunk_repr space mismatch** — Jina embeddings live in a different
  representation space from Qwen's internal hidden states.
  Retrieval may pick chunks that score high in Jina-space but are
  not the most useful for Qwen's attention.
- **Linear-attn v1 re-forward** is correct but undercount the
  speedup of ReAttention. v2 caching is the real ask.
- **MAGS calibration data quality** — for the contrastive `T-`
  trajectory to be informative, the "wrong" chunks need to be
  *plausibly mis-retrievable*, not pure noise. The current
  bottom-K-by-cosine selection may pick chunks too unlike the
  query to expose the failure mode MAGS targets.
- **`attn_output_gate=true`** in Qwen3.5 — the gate is preserved by
  the patched forward (we copy that step verbatim), but MAGS
  intervenes on the *block-output residual*, not on the gated
  attention output. If the gate is what causes drift, MAGS won't
  see it. Could be worth a per-head intervention as a v2 of MAGS.

## 9. Useful one-liners

```bash
# Re-score saved results with the fixed scorer
python3 scripts/rescore.py data/niah/results_4k_full.jsonl

# Run only ReAttention (faster than baseline at 4K)
python3 scripts/03_run_niah.py --cases data/niah/niah_4k.jsonl \
    --out /tmp/r.jsonl --modes reattn --limit 5

# Inspect MAGS fire rate on specific cases
python3 scripts/05_inspect_mags_fire.py
```
