# sprag — design notes & status

> Last updated: 2026-05-25 (end of CPU-side scaffolding session).

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
   real value of the approach.
3. **RULER multi-needle / multi-key** — single-needle is too easy.
   Multi-needle / multi-value forces cross-chunk integration, where
   semantic drift (and therefore MAGS) is more likely to matter.
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
