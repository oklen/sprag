# sprag — design notes & status

> Last updated: 2026-05-25 (GPU-bringup session: Tesla T4, fp16 + SDPA).

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
