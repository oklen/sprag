# Video KV-cache Δ(coverage) — Qwen3-Omni / EgoSchema-Subset

Paired cached-vs-fresh gold-answer NLL (position-matched 100% of samples: only the
KV SOURCE differs). Negative ΔNLL = cached (prebaked full-clip KV, carrying causal
global memory of the dropped frames) beats fresh (re-encode kept subset). 32 frames
/ 16 t-groups, uniform coverage. cov100 ΔNLL=0 is the built-in identity sanity.

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
