# Claims ledger — every blog claim → status → evidence → source

Status: ✅ solid (significant, reproduced) · ⚠️ open (directional / underpowered) ·
🚫 ruled-out (a competing claim we falsified). Use only ✅ as headline claims; frame
🚫 as refutations of alternatives; keep ⚠️ in the open-problem section.

| # | Claim | Status | Evidence (n, stat) | Source |
|---|---|---|---|---|
| 1 | Reusing a full-context KV-cache on a subset beats recompute (memory bonus) | ✅ | EgoSchema n=500 ΔNLL −0.039@cov20 p<1e-4, +Δacc; fp32 n=100 confirms | `docs/OMNI_RESULTS.md`; `data/omni_cov_full,fp32` |
| 2 | Bonus is monotone in coverage, vanishes at full coverage (identity gate) | ✅ | curves §A/§B; cov100 ΔNLL=0 exact in every run | `figure_data.md §A,§B` |
| 3 | **Cross-modal associative recovery** (audio trace in video KV) | ✅ | Video-MME n=597, cov100 −0.070 p<1e-4; vision baseline +0.000; mode-invariant gap | `docs/OMNI_RESULTS.md`; `data/omni_vm_xrecover*` |
| 4 | Bonus largest when recompute most starved (omit-bridge amplifies 2–4×; longer clips bigger) | ✅ | center-mode > uniform every cov p<1e-4; VideoMME −0.23 ≫ Ego −0.039 | `docs/OMNI_RESULTS.md` E4 |
| 5 | Accuracy lift, not just NLL | ✅ | short Video-MME +18 pt acc @cov10; hotpotqa +9.5 pp z=2.4 | `experiments/omni_deepdive` #3; `figure_data.md §D` |
| 6 | Text full-attention couldn't surface it (answer adjacent to query); video could (temporal integration) | ✅ | text splice ≈ null; video significant | `docs/OMNI_RESULTS.md` finding 1 |
| 7 | **Scope boundary**: holds only for unified-context + subset-inference; corpus-RAG asymmetry=0 | ✅ (by construction) | mechanism + cov100 identity; no accuracy claim for generic RAG | README §4 |
| 8 | "cached ≥ fresh" is **not universal** — text low-coverage cliff (cache worse) | 🚫→✅ | 3-arm n=231: c0 gap +0.5 NLL (cliff), crossover ~c25–c50 | `experiments/cov_curve`; `figure_data.md §A` |
| 9 | The cliff is keep-set **starvation**, not the position convention | ✅ | origpos +0.519 ≈ compact +0.571 @c0, diff <1 SEM (n.s.) | `experiments/cov_curve` |
| 10 | "Position-preserving ≫ compaction / attack on ReKV" | 🚫 NULL | penalty +0.006 n.s. even at t_grid=32 cov10; no temporal/long concentration | `experiments/omni_deepdive` #3; `figure_data.md §F` |
| 11 | Faithful ReKV ≈ ours (tie, not a win we claim) | ✅ | rekv +0.009 ns vs ours (cov20) | `docs/OMNI_RESULTS.md` baselines; `figure_data.md §G` |
| 12 | MuKV win = fine token selection (free compaction), largely query-driven | ✅ | decomp +0.0152 vs +0.0025; query-free helps long only | `docs/OMNI_RESULTS.md` Session-5 |
| 13 | "Cross-modal trace lives in deep layers" | 🚫 mislocalized | early–mid L8–24: first-8 = 68% of gap, plateau L24 (n=252) | `experiments/omni_deepdive` #2; `figure_data.md §E` |
| 14 | "Sink-dup harm = attention dilution" | 🚫 falsified | it's a decode-trajectory failure (open `<think>`); fixed | `experiments/coverage_sinkdup` |
| 15 | Double-edged trace: helps 2-hop, misleads multi-hop | ⚠️ OPEN | hotpotqa +9.5pp z=2.4 (sig); musique −6.8pp z=−1.3 (n.s.); 2wikimqa null ⇒ task-structure not hop-count; coverage is positional not semantic | `figure_data.md §D`; Path B running |

## Hard caveats to honor in the writing
- **Do not** claim an accuracy win for generic retrieval-RAG (#7).
- **Do not** claim "position-preserving beats compaction" (#10) or "we beat ReKV by
  keeping positions" — it's a tie (#11).
- **Do not** state the double-edged/hop claim as a result (#15) — it is an open
  problem; the only defensible positive is hotpotqa.
- Cross-modal wording: "Cross-Modal Pattern Completion via Contextualized KV
  Imprinting" / "Modality-Absent Associative Recall" — and note the audio KV is
  *physically removed*, not masked (verified in `44_omni_coverage.py`
  DROP_AUDIO_AT_USE), so recovery comes purely from the video-token KV imprint.

## Models / setups (for the methods paragraph)
- **Text:** Qwen3-30B-A3B-Instruct-2507 (+ Qwen3.5-27B thinking for the sink-dup diag),
  48-layer full-attn MoE; LongBench 2wikimqa/hotpotqa/musique; chunk 256; KV splice at
  HF cache level.
- **Video / cross-modal:** Qwen3-Omni-30B-A3B Thinker (48-layer full-attn, M-RoPE);
  EgoSchema-Subset + Video-MME; position-preserving M-RoPE splice; paired gold-answer
  NLL, Wilcoxon; cov100 identity gate; bf16 with fp32 control.
