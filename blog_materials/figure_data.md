# Figure data — plot-ready tables

All gaps are **paired** (same items, only the KV source differs). NLL = gold-answer
negative log-likelihood (text: mean log-PPL); **negative = cached/reuse wins**.
`cov100` is the built-in identity gate (reuse == recompute).

---

## §A — HERO: text coverage curve, 3-arm (A3B-Instruct, LongBench, n=231)

`experiments/cov_curve/` · `scripts/33_origpos_3arm.py` · `data/a3b_cov_3arm.s*`.
Gap vs fresh (mean log-PPL). Shows the **cliff** at c0 and exact convergence at c100.

| coverage | n | NLL fresh | origpos − fresh (SEM) | compact − fresh (SEM) | acc f / o / c |
|---:|---:|---:|---:|---:|---:|
| c0   | 231 | 13.675 | +0.519 (.122) | +0.571 (.118) | .68 / .68 / .69 |
| c25  | 231 | 14.011 | −0.042 (.105) | +0.101 (.096) | .67 / .71 / .70 |
| c50  | 211 | 13.175 | −0.341 (.091) | −0.211 (.088) | .66 / .68 / .67 |
| c75  | 206 | 12.627 | −0.093 (.056) | −0.021 (.054) | .66 / .67 / .69 |
| c100 | 181 | 12.388 | +0.019 (.011) | +0.019 (.011) | .67 / .66 / .66 |

Notes for the plot: shade c0→~c25 as the "degeneration / cliff" regime, c25→c100 as
the "memory-bonus" regime. origpos and compact are **bit-identical at c100** (both
+0.019) — that's the identity gate. origpos sits at/below compact everywhere (the
mid-range convention edge; modest).

## §B — HERO overlay: video coverage curve (Qwen3-Omni, Video-MME)

Monotone, **no cliff** (cov10 already wins). Two series usable:

**B1. Vision-only, n=597** (`omni_vm_vision*`):

| cov | ΔNLL (cached − fresh) | SEM | p |
|---:|---:|---:|---:|
| 20  | −0.2087 | .0155 | <1e-4 |
| 40  | −0.0892 | .0098 | <1e-4 |
| 60  | −0.0423 | .0065 | <1e-4 |
| 80  | −0.0134 | .0055 | 0.002 |
| 100 | +0.0000 | 0 | — |

**B2. M-RoPE hard rerun, n=236** (`experiments/omni_deepdive`, the accuracy headline):

| subset | ours − fresh @cov10 (NLL) | acc fresh→ours @cov10 |
|---|---:|---|
| ALL (n=236) | −0.229 | .34 → .42 (**+8 pt**) |
| Short videos (n=87) | −0.325 | .33 → .51 (**+18 pt**) |

**B3. EgoSchema vision-only, n=500** (hardened core, fp32-confirmed):

| cov | ΔNLL | SEM | p | Δacc |
|---:|---:|---:|---:|---:|
| 20  | −0.0392 | .0053 | <1e-4 | +.012 |
| 40  | −0.0242 | .0039 | <1e-4 | +.020 |
| 60  | −0.0175 | .0027 | <1e-4 | +.010 |
| 80  | −0.0044 | .0019 | 0.025 | +.004 |
| 100 | 0 | 0 | — | 0 |

(Crossover c*≈0.8; reproduced in fp32 n=100 ⇒ not a bf16 artifact.)

## §C — HOOK: cross-modal associative recovery (Video-MME, n=597)

`docs/OMNI_RESULTS.md`. Prebake-with-audio → drop-audio-at-use. The **gap** row is the
pure audio trace (xrecover minus vision-only on identical clips).

| cov | xrecover ΔNLL | vision-only ΔNLL | gap = audio trace | xrecover Δacc |
|---:|---:|---:|---:|---:|
| 20  | −0.3087 | −0.2087 | −0.100 | +0.069 |
| 40  | −0.2179 | −0.0892 | −0.129 | +0.059 |
| 60  | −0.1418 | −0.0423 | −0.099 | +0.008 |
| 80  | −0.0945 | −0.0134 | −0.081 | −0.010 |
| 100 | **−0.0701** | **+0.0000** | **−0.070** | +0.008 |

Headline annotation: at cov100 vision-only is exactly 0 (identity); the audio-prebaked
cache is −0.070 (p<1e-4, n=597) — the only difference is whether audio was present at
*prebake*. Center-mode (omit-bridge) gives the same gap (mode-invariant) ⇒ genuine
cross-modal signal.

## §D — Open-problem: by-dataset accuracy gap (A3B-Instruct, Δacc = cached − fresh)

`experiments/coverage_sinkdup` / `scripts/34_a3b_diag.py` + `data/a3b_cov_fix.s*`.
z = mean/SEM of the paired per-item acc difference. **Use for the open-problem panel.**

| dataset (hops) | c0 | c25 | c50 | c75 | c100 |
|---|---|---|---|---|---|
| hotpotqa (2-hop)  | +.054 (z1.27) | **+.095 (z2.41)** | +.048 (z1.35) | **+.063 (z2.05)** | 0 |
| 2wikimqa (2-hop)  | +.024 (z0.58) | +.012 (z0.38) | −.013 (z−0.30) | 0 (z0) | −.028 (z−1.42) |
| musique (multi)   | **−.068 (z−1.30)** | −.027 (z−0.63) | −.014 (z−0.38) | +.029 (z0.70) | −.017 (z−0.57) |

Reading: hotpotqa significantly positive (the bonus where the task benefits); musique
negative at low coverage but **n.s.**; 2wikimqa null ⇒ sign tracks task structure, not
hop-count. (Note: coverage here is *positional*, not semantic — see README §8.)

musique keep-set sizes (why c0 is so starved): answer chunk depth median 16 (of up to
44) 256-tok chunks; chunks kept = **0** @c0 (answer chunk only), ~4 @c25, ~9 @c50,
~13 @c75, ~18 @c100.

## §E — Layer-wise: where the cross-modal trace is read out (n=252, cov100)

`scripts/48_omni_layerwise.py`. Cumulative-depth swap; full gap = −0.100 NLL (70% of
records cached-better). Trace is **early–mid**, not deep.

| layers swapped to cached [0,d) | fraction of full gap |
|---:|---:|
| first 4  | 36% |
| first 8  | **68%** |
| first 12 | 72% |
| first 24 | **100%** (plateau) |

## §F — Ruled-out: M-RoPE compaction penalty is NULL (n=236)

`experiments/omni_deepdive` #3. A real grid-shear would concentrate on temporal/long;
it does the opposite ⇒ noise.

| subset | compact − ours @cov10 (SEM) | t | concentrates as predicted? |
|---|---:|---:|---|
| ALL (n=236) | +0.0061 (.0030) | ~2.0 | — |
| Temporal Qs (n=27) | +0.0056 (.0060) | ~0.9 | ❌ smaller, n.s. |
| Long videos (n=65) | +0.0078 (.0042) | ~1.9 | ❌ |
| Short videos (n=87) | +0.0104 (cov20) | ~1.6 | ❌ as large as long |

## §G — Baselines: faithful ReKV / MuKV vs ours (EgoSchema, Δ vs ours, cov20)

`scripts/47_omni_baselines.py`. Negative = beats `ours` (position-preserving reuse).

| arm | uniform n=500/101 | read |
|---|---|---|
| fresh | +0.026** | recompute loses |
| ours_compact | +0.0065* | compaction ≈ free |
| rekv (faithful InfLLM) | +0.009 ns | **ties** ours |
| mukv (token-select @ orig pos) | **−0.021*** | informed selection is the one lever |

MuKV decomposition (n=500): token→group coarsening costs +0.0152 (≈55% of gain);
group→compact costs +0.0025 (free). Query-axis (cov20, deployable=query-free):
`mukv_self` EgoSchema −0.0117 (ns) / Video-MME −0.066 (p2e-3); `mukv_fft` useless.
⇒ MuKV's win is **fine token selection, largely query-driven**; ours is the right
prebakeable default.
