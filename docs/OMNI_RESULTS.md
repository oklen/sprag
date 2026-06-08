# Video KV-cache Δ(coverage) — Qwen3-Omni / EgoSchema-Subset

Paired cached-vs-fresh gold-answer NLL (position-matched 100% of samples: only the
KV SOURCE differs). Negative ΔNLL = cached (prebaked full-clip KV, carrying causal
global memory of the dropped frames) beats fresh (re-encode kept subset). 32 frames
/ 16 t-groups, uniform coverage. cov100 ΔNLL=0 is the built-in identity sanity.

## EXECUTIVE SUMMARY — does cached KV reuse beat fresh recompute in a video LLM, and is there cross-modal memory?

Setup: Qwen3-Omni-30B-A3B Thinker (48-layer full-attn MoE), position-preserving M-RoPE
KV-cache splice. "cached" = reuse the prebaked whole-clip KV at original 3D positions,
keeping c% of video t-groups; "fresh" = re-encode only the kept subset at the SAME
positions. Metric = paired gold-answer NLL (cached−fresh); negative = cached wins.
cov100 ΔNLL=0 is the built-in identity sanity (reuse==recompute at full coverage).

**FOUR HEADLINE FINDINGS**

1. **Global-context bonus is real (cached BEATS fresh).** Vision-only, EgoSchema n=500:
   ΔNLL −0.039(cov20)→0(cov100), all low/mid coverages p<1e-4, +Δacc; reproduced in
   fp32 (n=100) ⇒ not a bf16 kernel artifact. Crossover c*≈0.8 (bonus vanishes by ~80%
   coverage). The prebaked KV carries causal global memory of the dropped frames that a
   fresh subset-recompute lacks. (The text full-attention experiment could NOT surface
   this — there the answer sat adjacent to the query; temporal video QA gives the global
   memory something to contribute.)

2. **Omit-bridge (center-window) AMPLIFIES it ~2–4×.** When fresh keeps a contiguous
   local window (misses the rest of the timeline) the bonus grows at every coverage
   (EgoSchema cov80 −0.016 vs −0.004 uniform; Video-MME cross-modal confirms the same).
   Direct evidence the cached cache supplies reasoning context the local window cannot
   assemble.

3. **TRUE cross-modal associative recovery — the headline novelty (n=597, p<1e-4).**
   Prebake the clip WITH audio (video KV absorbs audio↔visual associations), then at
   use-time DROP all audio and keep c% video; cached(video KV carrying the audio trace)
   vs fresh(video re-encoded with NO audio). **At cov100 ΔNLL=−0.070 (p<1e-4)** while the
   audio-free vision baseline is exactly 0 — all frames kept at identical positions, the
   ONLY difference is whether the video KV was baked with audio. cached<fresh ⇒ the video
   tokens' KV durably stored audio information, recoverable with audio entirely absent at
   inference. The xrecover−vision gap (−0.07…−0.13 at every coverage, mode-invariant under
   center vs uniform) isolates this as a genuine cross-modal signal, not a coverage
   artifact. This is the differentiator vs ReKV (sliding-window, local only) and MuKV
   (compression): a prebaked cache that holds cross-modal associative memory.

4. **Counter-intuitive: always-on audio SHRINKS the reuse bonus (~3–4×).** When audio is
   instead kept in BOTH arms (Video-MME ±audio), the bonus shrinks — the reuse bonus is an
   INFORMATION-STARVATION effect (largest when fresh is most context-starved); a redundant
   always-available modality compensates for dropped frames so global-video memory adds
   less. The same mechanism explains why DROPPING audio at use (finding 3) makes the cached
   trace the sole audio source ⇒ cached ≫ fresh. (Video-MME vision-only bonus −0.231 ≫
   EgoSchema −0.039: longer/richer clips starve fresh far more.)

**Scale/precision:** cross-modal recovery scaled n=87→597 (cov100 −0.088→−0.070, ~3×
tighter SEM, all p<1e-4); center-mode cov100 byte-identical to uniform ⇒ pipeline
deterministic. Engine fp32 identity gate PASS (3.24e-5). All on Qwen3-Omni-30B, bf16
(floor cancels across the paired arms). Branch video-kv-omni.

**Deep-dive corrections** (see [`../experiments/omni_deepdive`](../experiments/omni_deepdive)):
the cross-modal trace (finding 3) is read out in **early–mid layers (~L8–24)**, not deep
ones; and position-preserving reuse vs compaction is a **NULL** even under maximal M-RoPE
shear — we do **not** claim "repositioning hurts." The robust, unified effect is the
`cached/ours > fresh` global-memory bonus, reinforced at +18 pt acc on short Video-MME clips.

---

## bf16, n=500 (omni_cov_full.json) — HARDENED CORE
| cov | ΔNLL | SEM | Wilcoxon p | % cached better | acc_cached | acc_fresh | Δacc |
|----:|-----:|----:|-----------:|----------------:|-----------:|----------:|-----:|
|  20 | −0.0392 | .0053 | **<1e-4** | 64.0% | .684 | .672 | +.012 |
|  40 | −0.0242 | .0039 | **<1e-4** | 59.8% | .694 | .674 | +.020 |
|  60 | −0.0175 | .0027 | **<1e-4** | 61.6% | .690 | .680 | +.010 |
|  80 | −0.0044 | .0019 | **0.025**  | 54.2% | .710 | .706 | +.004 |
| 100 |  0.0000 | 0    | —          | —     | .702 | .702 |  0   |

## fp32 control, n=100 (omni_cov_fp32.json) — NOT a bf16 artifact
| cov | ΔNLL | SEM | Wilcoxon p | % cached better |
|----:|-----:|----:|-----------:|----------------:|
|  20 | −0.0266 | .0104 | **0.013** | 58% |
|  40 | −0.0138 | .0074 | 0.073     | 57% |
|  60 | −0.0120 | .0043 | **0.014** | 64% |
|  80 | −0.0039 | .0043 | 0.114     | 57% |
| 100 |  0.0000 | 0    | —          | —  |

**Conclusion.** The global-context bonus (cached KV beats fresh subset-recompute) is
real, statistically significant, and monotone in coverage — strongest at low
coverage, vanishing at full coverage. It appears in BOTH perplexity and accuracy at
n=500, and survives fp32 (so it is not a bf16 prefill-vs-decode kernel artifact; that
floor cancels across arms anyway). This is the effect the text full-attention
experiment could NOT surface (there the answer was adjacent to the query); EgoSchema's
temporal reasoning gives the prebaked global memory something to contribute.

Vision-only. Next: cross-modal (audio+video) arm; E4 omit-bridge stress test.

## E4 omit-bridge stress test — center-mode coverage, bf16 n=500 (omni_cov_e4.json)
Same protocol but kept frames = a CONTIGUOUS window (center) instead of uniform —
so fresh sees only a local segment and MISSES the rest of the timeline ("bridge"),
while the cached KV still remembers the whole clip.

| cov | ΔNLL (center) | (uniform) | Wilcoxon p | % cached better | Δacc |
|----:|--------------:|----------:|-----------:|----------------:|-----:|
|  20 | −0.0567 | −0.0392 | <1e-4 | 68.2% | +.020 |
|  40 | −0.0437 | −0.0242 | <1e-4 | 67.2% | +.026 |
|  60 | −0.0276 | −0.0175 | <1e-4 | 64.2% | +.018 |
|  80 | −0.0164 | −0.0044 | <1e-4 | 62.8% | +.022 |
| 100 |  0.0000 |  0.0000 | —      | —     |  0   |

**Finding:** the omit-bridge regime AMPLIFIES the bonus at every coverage (e.g. cov80
−0.016 vs −0.004, ~4×), with Δacc now a steady +2pp. The cached cache demonstrably
supplies reasoning context the fresh local-window assembly lacks — strongest evidence
yet that cached can beat fresh.

## Cross-modal: Video-MME ±audio, same short clips, n=87 (uniform coverage)
Audio = separate ALWAYS-KEPT stream (present in both arms, capped 120s); only the
video KV differs (cached global vs fresh subset). So ΔNLL = value of global VIDEO
memory, with vs without audio present.

| cov | ΔNLL vision-only | p | ΔNLL audio+video | p |
|----:|-----------------:|--:|-----------------:|--:|
|  20 | −0.2312 | <1e-4 | −0.0638 | 0.14 |
|  40 | −0.0862 | 0.002 | −0.0362 | 0.019 |
|  60 | −0.0599 | 0.005 | −0.0304 | 0.048 |
|  80 | +0.0039 | ns    | −0.0014 | ns |
| 100 |  0      | —     |  0      | — |

**Finding (counter to the naive hypothesis).** Adding audio does NOT amplify the
reuse bonus — it SHRINKS it ~3-4×. The bonus is an information-starvation effect:
it is largest when the fresh subset is most context-starved. An always-available
complementary modality (audio) compensates for the dropped video frames, so the
fresh arm is less starved and the cached global-video memory adds less. Also note
Video-MME vision-only bonus (cov20 −0.231) ≫ EgoSchema (−0.039): longer/richer
movie/TV clips make dropping frames hurt fresh far more. (n=87 short clips; audio
arm SEM large; directionally consistent across cov20/40/60.)

## TRUE cross-modal associative recovery — Video-MME, n=87, prebake-with-audio + drop-audio-at-use
Prebake the clip WITH audio (video KV absorbs audio↔visual associations), then at
use-time DROP all audio tokens (+markers) and keep c% video. cached (video KV carrying
the audio trace) vs fresh (video re-encoded with NO audio), position-matched 100%.

| cov | ΔNLL xrecover | p | %c<f | Δacc | ΔNLL vision-only | gap (audio trace) |
|----:|--------------:|--:|-----:|-----:|-----------------:|------------------:|
|  20 | −0.3627 | <1e-4 | 77.0% | +.092 | −0.2312 | −0.131 |
|  40 | −0.2307 | <1e-4 | 72.4% | +.115 | −0.0862 | −0.145 |
|  60 | −0.1708 | <1e-4 | 72.4% | +.103 | −0.0599 | −0.111 |
|  80 | −0.1018 | 0.003 | 64.4% | +.057 | +0.0039 | −0.106 |
| 100 | −0.0878 | 0.001 | 66.7% | +.092 |  0.0000 | −0.088 |

**Headline: cov100 ΔNLL = −0.088 (p=0.001).** At full coverage all video frames are
kept at identical positions; the ONLY difference is whether the video KV was prebaked
with audio. cached < fresh => the video tokens' KV absorbed usable audio information
recoverable with audio entirely absent at use time = TRUE cross-modal associative
recovery. The gap (xrecover − vision) ≈ −0.09…−0.14 at every coverage is the audio
trace ON TOP of the vision global-memory bonus; Δacc +6–12pp.

Consistent with the always-on-audio result (audio shrinks the marginal bonus because
it is redundant with cached video memory): when audio is instead DROPPED at use, the
cached video KV's absorbed audio trace is the sole audio source -> cached ≫ fresh.
(n=87 short clips, bf16; scale n to harden.)

## SCALED cross-modal associative recovery — Video-MME 12-chunk (570 mp4s, short filter)
Same protocol as the n=87 section (prebake-with-audio → drop-audio-at-use, video KV
carries the audio trace; cached vs fresh-no-audio, position-matched). Scaled from the
n=87 pilot to **n=597 unique clips** (Video-MME chunks 01–12), worker 3878333.

**XRECOVER (n=597, bf16):**

| cov | ΔNLL (cached−fresh) | SEM | wilcox_p | %c<f | acc_c | acc_f | Δacc |
|----:|--------------------:|----:|---------:|-----:|------:|------:|-----:|
|  20 | −0.3087 | 0.0186 | <1e-4 | 76.9% | 0.472 | 0.404 | +0.069 |
|  40 | −0.2179 | 0.0152 | <1e-4 | 73.0% | 0.514 | 0.456 | +0.059 |
|  60 | −0.1418 | 0.0133 | <1e-4 | 67.3% | 0.501 | 0.492 | +0.008 |
|  80 | −0.0945 | 0.0116 | <1e-4 | 62.5% | 0.497 | 0.508 | −0.010 |
| 100 | **−0.0701** | 0.0106 | **<1e-4** | 60.1% | 0.506 | 0.497 | +0.008 |

**Headline holds at scale: cov100 ΔNLL = −0.070, p<1e-4 (n=597).** The pilot's
−0.088 (p=0.001, n=87) reproduces with ~3× tighter SEM (0.0106 vs 0.0298). At full
visual coverage the only difference is whether the video KV was prebaked with audio;
cached < fresh for 60.1% of clips ⇒ the video tokens' KV durably absorbed audio
information recoverable with audio absent at use time = true cross-modal associative
recovery. Monotonic decay (−0.31 → −0.07 as coverage 20→100) is smooth and every
coverage is p<1e-4.

**VISION-only baseline (n=597, same 570-mp4 clips, no audio):**

| cov | ΔNLL (cached−fresh) | SEM | wilcox_p | %c<f | acc_c | acc_f | Δacc |
|----:|--------------------:|----:|---------:|-----:|------:|------:|-----:|
|  20 | −0.2087 | 0.0155 | <1e-4 | 71.2% | 0.436 | 0.399 | +0.037 |
|  40 | −0.0892 | 0.0098 | <1e-4 | 66.5% | 0.472 | 0.449 | +0.023 |
|  60 | −0.0423 | 0.0065 | <1e-4 | 60.1% | 0.476 | 0.472 | +0.003 |
|  80 | −0.0134 | 0.0055 |  0.002 | 57.3% | 0.486 | 0.491 | −0.005 |
| 100 | **+0.0000** | 0.0000 | — | 0.0% | 0.482 | 0.482 | +0.000 |

**PAIRED (n=597, identical clips) — the audio-trace gap:**

| cov | xrecover ΔNLL | vision-only ΔNLL | gap = pure audio trace |
|----:|--------------:|-----------------:|-----------------------:|
|  20 | −0.3087 | −0.2087 | −0.100 |
|  40 | −0.2179 | −0.0892 | −0.129 |
|  60 | −0.1418 | −0.0423 | −0.099 |
|  80 | −0.0945 | −0.0134 | −0.081 |
| 100 | **−0.0701** | **0.0000** | **−0.070** |

**Headline at scale (n=597).** At cov100 vision-only is exactly 0 (reuse == recompute,
identity sanity), while the audio-prebaked cache is −0.0701 (p<1e-4) — the video KV
durably absorbed audio information recoverable with audio absent at use = true
cross-modal associative recovery, now 6.9× the pilot's n=87 with ~3× tighter SEM.
The −0.07…−0.13 gap at every coverage is the audio trace layered on top of the
vision global-memory bonus. Vision-only alone also reproduces the global-context
bonus (cov20 −0.209 → 0, monotonic, all p<1e-4). Worker 3878333, bf16, 4-shard.

## E4 CENTER-MODE cross-modal recovery — Video-MME, n=597 (COVERAGE_MODE=center)
Same n=597 clips as the scaled uniform run, but fresh keeps a CONTIGUOUS center window
(not uniform spread) → fresh sees only a local segment, maximally context-starved.
Tests whether the audio-trace recovery amplifies the way the vision-only bonus does.
omni_vm_{xrecover,vision}_e4.json. Worker 3878333, bf16, 8-shard (GPU0-7).

| cov | xrecover-center ΔNLL | vision-center ΔNLL | gap = audio trace |
|----:|---------------------:|-------------------:|------------------:|
|  20 | −0.3443 (p<1e-4, 81.1%) | −0.2532 (p<1e-4) | −0.091 |
|  40 | −0.2633 (p<1e-4) | −0.1561 (p<1e-4) | −0.107 |
|  60 | −0.1699 (p<1e-4) | −0.0785 (p<1e-4) | −0.091 |
|  80 | −0.1201 (p<1e-4) | −0.0284 (p<1e-4) | −0.092 |
| 100 | **−0.0701** (p<1e-4) | **+0.0000** (identity) | −0.070 |

Δacc (xrecover-center): cov20 +8.5pp, cov40 +5.5pp, cov60 +2.0pp.

**vs UNIFORM (amplification):** center grows the global-memory bonus in BOTH arms.
Vision: cov40 −0.089→−0.156 (1.75×), cov60 −0.042→−0.079 (1.86×), cov80 −0.013→−0.028
(2.1×, now p<1e-4 vs p=.002). Xrecover: cov40 −0.218→−0.263, cov60 −0.142→−0.170,
cov80 −0.095→−0.120 (~1.2–1.3×). Mechanism confirmed on Video-MME + cross-modal: a
contiguous local window starves fresh more, so the cached whole-clip KV contributes more.

**KEY: the pure audio-trace gap (xrecover − vision) is ~MODE-INVARIANT** (−0.09…−0.11
mid-coverages vs uniform's −0.08…−0.13; cov100 −0.070 in both). The center-vs-uniform
amplification cancels in the gap because visual-frame starvation hits both arms equally
⇒ the audio trace is a genuine cross-modal (audio↔visual association in the KV) signal,
NOT a visual-coverage artifact. cov100 reproduced byte-identical to the uniform run
(−0.0701/+0.0000, same SEM) since center==uniform at full coverage ⇒ pipeline determinism.


## FAITHFUL ReKV / MuKV BASELINE COMPARISON — Qwen3-Omni / EgoSchema-Subset

Engine: `scripts/47_omni_baselines.py`. All arms start from the SAME full prebaked
KV (one forward; hooks capture pre-RoPE post-norm Q/K/V). KEYSTONE GATE: re-rotating
captured K at its original M-RoPE positions reproduces the model's cached K to
**0.00e+00 over all 48 layers** (V is position-independent, also exact) — so InfLLM
repositioning (re-rotate at NEW positions, reusing the model's own
`apply_rotary_pos_emb`) is exact. Arms differ on TWO axes only:

| arm | selection of kept video t-groups | positions |
|-----|----------------------------------|-----------|
| fresh | uniform/center subset (RE-ENCODED, reference) | gapped |
| **ours** | uniform/center | ORIGINAL gapped (position-preserving reuse) |
| ours_compact | uniform/center | COMPACTED (gaps removed) |
| rekv_origpos | per-layer query-retrieval (sink+local+top-k) | ORIGINAL |
| **rekv** | per-layer query-retrieval | COMPACTED (full InfLLM ReKV) |
| **mukv** | per-layer dual-signal (attn+FFT) token selection | ORIGINAL |

ReKV faithfully reproduces its two signatures: per-layer pre-RoPE query·key block
retrieval (mean-pooled block reps, GQA-expanded; `_calc_block_topk`) + sink(n_init)
+ local window, then InfLLM compaction. MuKV reproduces its dual-signal token score
`α·Î_att+(1−α)·Î_fft` (α=0.7; attn = question-query·key mass, fft = mean |rFFT(k)|),
keep top-budget tokens. Matched budget = c% of t-groups (MuKV: c% of video tokens).
Metric = paired gold-answer NLL (lower = better). cov100 = all groups kept =>
compact==original==full forward (built-in identity; PASS, all arms bit-identical).

### uniform coverage, n=100 (Δ vs ours, paired Wilcoxon)
| cov | fresh | ours_compact | rekv | mukv |
|----:|------:|-------------:|-----:|-----:|
|  20 | +0.026** | +0.0065* | +0.0091 | **−0.021*** |
|  40 | +0.010 | −0.000 | −0.004 | −0.011 |
|  60 | +0.012* | −0.001 | +0.003 | +0.001 |
|  80 | −0.002 | −0.001 | −0.001 | −0.000 |
| 100 | 0 (identity) | 0 | 0 | 0 |

### center coverage (E4 omit-bridge), n=96
| cov | fresh | ours_compact | rekv | mukv |
|----:|------:|-------------:|-----:|-----:|
|  20 | +0.040** | +0.0094*** | +0.001 | **−0.033*** |
|  40 | +0.042** | +0.0059* | −0.005 | −0.015 |
|  60 | +0.019** | +0.003 | −0.002 | −0.006 |
|  80 | +0.014** | +0.002 | −0.001 | −0.000 |
| 100 | 0 (identity) | 0 | 0 | 0 |

(* p<0.05, ** p<0.01, *** p<0.001; negative = beats ours)

### Four findings
1. **Position-preserving reuse beats fresh recompute** (the global-memory bonus),
   significant at low/mid coverage and **amplified + longer-lived in the omit-bridge
   regime** (center cov20–80 all p<0.01; fresh still +0.014 at cov80). Reproduces the
   core result against a recompute reference inside the unified engine.
2. **Repositioning costs ~nothing (and the small EgoSchema gap does NOT replicate).**
   `ours_compact` is marginally worse than `ours` here (+0.0065 uniform / +0.0094 center
   at cov20), but a dedicated stress-test designed to *maximize* the M-RoPE temporal
   shear — Video-MME long clips, `n_frames=64` (t_grid=32), aggressive cov 10/20/30,
   n=236 — finds this is a **NULL**: penalty ≤ +0.006 (t≈2 at cov10, n.s. elsewhere) and
   it does **not** concentrate on temporal questions or long videos (short videos pay as
   much) ⇒ noise, not real grid-shear. Qwen3-Omni's coarse integer temporal index is
   robust to T-axis compaction. **We do NOT claim "repositioning hurts / position-
   preserving ≫ compaction."** (Consistent with the Session-5 `+0.0025` decomposition
   below.) See [`../experiments/omni_deepdive`](../experiments/omni_deepdive) #3.
3. **Faithful ReKV nets to ≈ uniform reuse (does NOT beat ours).** Its per-layer
   query-retrieval gain (`rekv_origpos` slightly < ours, n.s.) is not recovered after
   its InfLLM repositioning, so full `rekv` ≈ `ours` everywhere. ReKV's
   sliding-window/retrieval design buys nothing over position-preserving reuse here —
   though note (finding 2) the repositioning itself is near-free, so the win is "reuse ≈
   retrieval", not "we beat them by keeping positions."
4. **Informed token selection (MuKV-style) is the ONE lever that beats uniform
   reuse** (cov20 −0.021 uniform / −0.033 center, p<0.01). Crucially our `mukv` arm
   keeps those tokens at ORIGINAL positions — i.e. the winner is *position preservation
   ⊕ informed selection*. ReKV (compacts) and vanilla `ours` (uniform) each have only
   one half; MuKV's informed selection is orthogonal to and composable with our
   position-preserving framework. (A fully-faithful MuKV that also compacts would pay
   the finding-2 penalty; our hybrid isolates its selection benefit.)

Net: against faithful ReKV + MuKV at matched budget, position-preserving full-KV
reuse dominates **recompute** and matches **ReKV** (repositioning is near-free, so this
is a tie not a win), and the remaining headroom is informed selection — which our
framework absorbs without giving up true positions. n≈100 EgoSchema
(101 subset videos on CephFS; HF chunk re-download for n=500 was rate-limited).

---

## SESSION-5: Selection-vs-Compression @ scale + the Query-aware/agnostic axis (2026-06-07)

Engine `scripts/47_omni_baselines.py` extended with group-level MuKV arms and three
token-level query-signal variants. All arms share ONE prebake; cov100 = all arms
bit-identical (built-in identity gate, PASS in every run below). Paired gold-NLL Δ
vs `ours` (position-preserving uniform reuse), Wilcoxon p, at cov20 (most aggressive
eviction; effects shrink to ns by cov80).

### (A) What is MuKV's advantage — selection or compression?
Arms: `mukv` (token-level select @ orig pos), `mukv_grp` (group-level select @ orig),
`mukv_compact` (group-level select @ compact pos).

| condition | mukv (token) | mukv_grp (group) | mukv_compact | read |
|---|---|---|---|---|
| EgoSchema uniform n=101 | −0.0208 (p=.044) | +0.0004 ns | −0.0002 ns | token only |
| EgoSchema center  n=100 | −0.0343 (p=.005) | −0.0134 ns | −0.0140 ns | token only |
| **EgoSchema n=500** | **−0.0278 (p=6e-11)** | −0.0126 (p=1.4e-3) | −0.0101 (p=.02) | token≫group |
| Video-MME n=160 | −0.1800 (p=3e-15) | −0.0553 | −0.0558 | token≫group |

**Decomposition (n=500):** token→group coarsening costs **+0.0152** (≈55% of the gain);
group orig→compact (the compaction itself) costs only **+0.0025**. ⇒ **MuKV's advantage
is fine TOKEN-level selection; the position compaction it performs is essentially FREE**
(consistent with the keystone reposition-exact gate). Video-MME's long clips amplify all
effects ~6–9×.

### (B) Is the advantage query-aware? (deployability for prebake / RAG)
The premise of KV-cache reuse is *compute the context KV once, before the query, reuse
across many queries*. Query-aware compression (MuKV/ReKV score tokens against the question)
breaks that: the compressed cache can't be built until the query arrives. We isolate the
**query-signal axis** with three token-level arms (granularity=token, position=orig held
constant, only the scoring query changes):

- `mukv` — question-token query (query-AWARE; needs the query → not prebakeable)
- `mukv_self` — video tokens' own query, self-saliency (query-FREE, prebake-OK)
- `mukv_fft` — α=0, pure spectral magnitude (query-FREE, prebake-OK)
- `mukv_oracle` — 2nd prefill with the gold answer appended (oracle ceiling)

cov20 Δ vs `ours`:

| arm | query signal | EgoSchema n=120 | Video-MME n=160 | deployable |
|---|---|---|---|---|
| `mukv_oracle` | gold answer | −0.0259 (p=1.5e-3) | −0.1992 (p=6e-17) | ✗ |
| `mukv` | question | −0.0248 (p=2.6e-3) | −0.1800 (p=6e-15) | ✗ |
| `mukv_self` | video self | −0.0117 (p=.11 ns) | −0.0658 (p=2.3e-3) | ✓ |
| `mukv_fft` | spectral | +0.0031 (p=.77 ns) | +0.0138 (p=.03 worse) | ✓ |
| `ours` | none | — | — | ✓ |

**HEADLINE:** MuKV's headline win is **largely query-driven**. In the deployable
query-AGNOSTIC regime (the only valid one for prebaked reuse / RAG):
- **Short video (EgoSchema, ~3 min):** query-free selection gives **no** gain over
  position-preserving reuse `ours` (self ns, fft worse). The MuKV win is entirely query.
- **Long video (Video-MME):** content self-saliency (`mukv_self`) keeps a **modest but
  significant ~⅓** of the gain (−0.066, p=2e-3); FFT-only is useless/harmful.
- `mukv_oracle ≈ mukv` everywhere ⇒ query-based selection has a fixed ceiling that
  query-free signals cannot reach (it's not that our question-query is suboptimal).

⇒ **Position-preserving reuse is the right default for cache reuse; the only query-free
selection worth adding is self-saliency, and only on long content.** The large −0.18
numbers are query-aware "peeking" and are not realizable when the cache must be built
before the query.

Results JSONs on CephFS `/home/tiger/data/omni_bl_{ego_uni2,ego_ctr2,ego500,vmme,egoqax,vmmeqax}.json`.
Infra: all compute on clean worker 3888311 (309/310 GPU-locked by un-killable cross-PID-namespace
orphan procs from hung runners — a clean python EXIT frees GPU, a hang orphans ~70G/GPU).

### (B-addendum) Center-mode (omit-bridge) query-axis on Video-MME (n=160)

Re-ran the query-axis arms with COVERAGE_MODE=center (contiguous kept window — the
maximally context-starved eviction). **Importance-based arms (mukv/self/fft/oracle/
rekv/grp) are coverage-mode INVARIANT** — they select tokens by score, not by which
groups the mode picks, so their NLLs are byte-identical to the uniform run (also a
determinism check). Center only shifts the passive baselines: `ours` 4.077->4.130,
`fresh` 4.297->4.412 (contiguous window starves them more).

cov20 Δ vs `ours`, uniform vs center:

| arm | query signal | Δ uniform | Δ center |
|---|---|---|---|
| `mukv` | question (aware) | −0.180 (p=6e-15) | −0.234 (p=3e-18) |
| `mukv_oracle` | gold (ceiling) | −0.199 (p=6e-17) | −0.253 (p=2e-20) |
| `mukv_self` | video self (free) | −0.066 (p=2e-3) | **−0.119 (p=1.5e-6)** |
| `mukv_fft` | spectral (free) | +0.014 ns | −0.040 ns |

**Finding:** under harsher (center/omit-bridge) eviction the **deployable query-free
self-saliency advantage roughly doubles and becomes strongly significant** (−0.119,
p=1e-6) — the passive-reuse baseline degrades while score-based selection holds. FFT
stays useless. The query premium (mukv − mukv_self ≈ 0.114 nats) is mode-invariant.
⇒ query-free self-saliency is worth adding specifically in long-content + aggressive
contiguous-eviction regimes; otherwise position-preserving reuse remains the default.
Result: `/home/tiger/data/omni_bl_vmmeqax_ctr.json`.
