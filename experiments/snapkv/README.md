# SnapKV experiments on Qwen3-30B-A3B-Instruct

Attention-importance KV-cache compression (SnapKV) measured on the clean chat
pipeline (chat template + EOS, non-thinking `Qwen3-30B-A3B-Instruct-2507`, untruncated
greedy decode, `alias_match` accuracy) over LongBench-v1 multi-hop QA
(`2wikimqa`, `hotpotqa`, `musique`).

These are a **different axis** from the cache-*splice* coverage experiments
(`cached vs fresh` reuse fidelity): SnapKV asks *how much can importance-pruning
compress the KV-cache before accuracy drops*.

## Runners

| file | what |
|---|---|
| `scripts/32_snapkv_coverage.py` | **SnapKV vs Fresh**. Observation window = the **question** (query-aware). Compresses the whole context. |
| `scripts/33_rag_snapkv.py` | **RAG-faithful, precomputable SnapKV**. anchor = top-retrieved (oracle) chunk, kept full + used as the observation window; non-anchor context compressed. Compares **Fresh / B(anchor-obs, precomputable) / A(question-obs, upper bound)** on the same records → the gap `A−B` is the *cost of being precomputable*. |

Implementation notes (both runners):
- `kvpress` is incompatible with transformers 5.7 → custom faithful SnapKV
  (per-kv-head Top-K + 1D max-pool kernel 7, GQA-aware).
- **No RoPE shift**: SnapKV keeps post-RoPE keys at their original positions, so
  relative distances stay exact (validated by a byte-exact keep-all identity gate).
- Prefill stays SDPA (eager would materialise an O(L²) attention matrix over the long
  context); only the *short* observation-window forward is run under an eager toggle so
  `output_attentions` is available — tiny memory.

Run (16-way sharded example): see `launchers/`.

## Results

### 1) SnapKV vs Fresh — question-obs, full LongBench (n=600)
`results/snapkv_cov.s*.json` (n=600); `results/snapkv_lim100/` is an earlier n=300 run.
**acc_fresh = 0.669**

| keep ratio | keep frac | acc_snap | Δ vs Fresh |
|---|---|---|---|
| 5%  | 0.053 | 0.580 | −0.089 |
| 10% | 0.103 | 0.602 | −0.067 |
| 20% | 0.203 | 0.625 | −0.043 |
| 30% | 0.302 | 0.637 | −0.032 |
| 50% | 0.502 | 0.654 | −0.015 |

Monotone, graceful degradation: ≈Fresh at 50% keep, within ~3–4 pts at 20–30%, and
~7–9 pts only at aggressive 5–10%. On these information-dense multi-hop contexts the
answer itself can be pruned, so low budgets hurt.

### 2) RAG-faithful (precomputable) SnapKV — anchor-obs vs question-obs (n=501)
`results/ragsnap_cov.s*.json`. **acc_fresh = 0.735**

| keep ratio | keep frac | acc_B (anchor, deployable) | ΔB | acc_A (question, upper bound) | ΔA | gap A−B |
|---|---|---|---|---|---|---|
| 5%  | 0.080 | 0.719 | −0.016 | 0.760 | +0.026 | +0.042 |
| 10% | 0.128 | 0.745 | +0.010 | 0.766 | +0.032 | +0.022 |
| 20% | 0.225 | 0.750 | +0.016 | 0.750 | +0.016 | +0.000 |
| 30% | 0.322 | 0.749 | +0.014 | 0.747 | +0.012 | −0.002 |
| 50% | 0.516 | 0.762 | +0.028 | 0.747 | +0.012 | −0.016 |

**Takeaway**: using the top-retrieved chunk as the observation window
(query-**independent** → precomputable) is essentially as good as the
non-precomputable query-aware version at keep ≥ 20% (gap ≈ 0), with only a ~4 pt
penalty at extreme 5%. Both stay ≥ Fresh because the oracle anchor keeps the answer in
cache; the question-aware window only helps preserve multi-hop *supporting* context
under very tight budgets (gap concentrated in `musique`).

**Deployment conclusion**: for precompute-RAG, SnapKV-compress each retrieved set
offline using the top-retrieved chunk as the observation window, keep ~20%+ →
near-lossless, no query needed at compression time.

## Caveats on comparability
- Experiment 2 **protects the answer** (anchor = answer chunk, always kept), so its
  curve is much flatter than Experiment 1 (where the answer may be pruned). Different
  record sets + anchor protection → the two are **not** directly comparable in absolute
  numbers; they answer different questions.
- SnapKV is a different axis from the cache-splice coverage experiments (reuse-vs-
  recompute fidelity) — don't put their Δ columns in one table.
