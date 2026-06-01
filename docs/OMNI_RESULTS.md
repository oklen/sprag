# Video KV-cache Δ(coverage) — first results (Qwen3-Omni / EgoSchema)

## Vision-only, n=100 (omni_cov_v1.json), 32 frames / 16 t-groups, uniform coverage
Paired cached-vs-fresh gold-answer NLL (position-matched: only the KV source differs).
Negative ΔNLL = cached (prebaked full-clip KV) better than fresh (re-encode kept subset).

| cov | ΔNLL  | SEM  | Wilcoxon p | % cached better | Δacc | acc_cached |
|----:|------:|-----:|-----------:|----------------:|-----:|-----------:|
|  20 | −0.0328 | .011 | **0.0033** | 63% | +.01 | .70 |
|  40 | −0.0104 | .007 | 0.133      | 53% | −.01 | .70 |
|  60 | −0.0117 | .004 | **0.0218** | 54% | −.01 | .70 |
|  80 | +0.0015 | .004 | 0.472      | 44% | +.01 | .71 |
| 100 |  0.0000 | 0    | —          | —   |  0   | .71 |

**Finding:** the prebaked full-clip KV carries causal global memory of the dropped
frames, giving significantly lower gold PPL than fresh subset-recompute at LOW
coverage (cov20 p=0.003), vanishing to 0 by cov~80 (crossover c* ≈ 0.8). This is
the "cached beats fresh" global-context bonus in PPL space — not seen in the text
full-attention experiment (answer was adjacent to the query there). Accuracy (~70%)
is too coarse at n=100 to move; PPL is the powered metric. cov100 ΔNLL=0 is the
built-in identity sanity (cached==fresh==full).

Caveats: vision-only; 100 samples (1 video chunk); bf16 (prefill-decode floor
cancels across arms since both score identically).

## Next
- Cross-modal arm (audio+video) — headline novelty; needs separate-audio-stream
  plumbing in the runner.
- E4 omit-bridge ("center" coverage) stress test.
- More samples; fp32 scoring control.
