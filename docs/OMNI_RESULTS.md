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
