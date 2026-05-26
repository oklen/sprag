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
