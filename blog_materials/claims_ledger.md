# Claims ledger — every blog claim → status → evidence → source

Status: ✅ solid (significant, reproduced) · ⚠️ open (directional / underpowered) ·
🚫 ruled-out (a competing claim we falsified). Use only ✅ as headline claims; frame
🚫 as refutations of alternatives; keep ⚠️ in the outlook section.

| # | Claim | Status | Evidence (n, stat) | Source |
|---|---|---|---|---|
| 1 | Reusing a full-context KV-cache on a subset beats recompute (memory bonus) | ✅ | EgoSchema n=500 ΔNLL −0.039@cov20 p<1e-4, +Δacc; fp32 n=100 confirms | `docs/OMNI_RESULTS.md`; `data/omni_cov_full,fp32` |
| 2 | Bonus is monotone in coverage, vanishes at full coverage (identity gate) | ✅ | curves §A/§B; cov100 ΔNLL=0 exact in every run | `figure_data.md §A,§B` |
| 3 | **Cross-modal associative recovery** (audio trace in video KV) | ✅ | Video-MME n=597, cov100 −0.070 p<1e-4; vision baseline +0.000; mode-invariant gap | `docs/OMNI_RESULTS.md`; `data/omni_vm_xrecover*` |
| 4 | Bonus largest when recompute most starved (omit-bridge amplifies 2–4×; longer clips bigger) | ✅ | center-mode > uniform every cov p<1e-4; VideoMME −0.23 ≫ Ego −0.039 | `docs/OMNI_RESULTS.md` E4 |
| 5 | Accuracy lift, not just NLL | ✅ | short Video-MME +18 pt acc @cov10; text: 2Wiki +1.1–2.3 pp ALL / drop_gold +3.4 pp, MuSiQue recovery cell +3.2 pp (§H) | `experiments/omni_deepdive` #3; `figure_data.md §H` |
| 6 | Text full-attention couldn't surface it (answer adjacent to query); video could (temporal integration) | ✅ | text splice ≈ null; video significant | `docs/OMNI_RESULTS.md` finding 1 |
| 7 | **Scope boundary**: holds only for unified-context + subset-inference; corpus-RAG asymmetry=0 | ✅ (by construction) | mechanism + cov100 identity; no accuracy claim for generic RAG | README Act IV |
| 8 | "cached ≥ fresh" is **not universal** — text low-coverage cliff (cache worse) | 🚫→✅ | 3-arm n=231: c0 gap +0.5 NLL (cliff), crossover ~c25–c50 | `experiments/cov_curve`; `figure_data.md §A` |
| 9 | The cliff is keep-set **starvation**, not the position convention | ✅ | origpos +0.519 ≈ compact +0.571 @c0, diff <1 SEM (n.s.) | `experiments/cov_curve` |
| 10 | "Position-preserving ≫ compaction / attack on ReKV" | 🚫 NULL | penalty +0.006 n.s. even at t_grid=32 cov10; no temporal/long concentration | `experiments/omni_deepdive` #3; `figure_data.md §F` |
| 11 | Faithful ReKV ≈ ours (tie, not a win we claim) | ✅ | rekv +0.009 ns vs ours (cov20) | `docs/OMNI_RESULTS.md` baselines; `figure_data.md §G` |
| 12 | MuKV win = fine token selection (free compaction), largely query-driven | ✅ | decomp +0.0152 vs +0.0025; query-free helps long only | `docs/OMNI_RESULTS.md` Session-5 |
| 13 | "Cross-modal trace lives in deep layers" | 🚫 mislocalized | early–mid L8–24: first-8 = 68% of gap, plateau L24 (n=252) | `experiments/omni_deepdive` #2; `figure_data.md §E` |
| 14 | "Sink-dup harm = attention dilution" | 🚫 falsified | it's a decode-trajectory failure (open `<think>`); fixed | `experiments/coverage_sinkdup` |
| 15 | Double-edged trace: helps 2-hop, misleads multi-hop (hop-count story) | 🚫 RESOLVED (superseded by #16–#20) | old §D numbers came from the artifact instrument; hop-count story dead | `figure_data.md §D` (historical) |
| 16 | Early "cache hurts multi-hop" was an **instrument artifact** (chain mode: answer para always kept → fresh = extractive oracle; alias-match penalized co-referent answers) | ✅ | redesigned uniform instrument w/ exact cov100 identity gate erases it (mq ΔNLL −0.002 @cov100, n=1500); case dumps show co-referent "wrong" answers (Tracy Mosby/McConnell) | `scripts/49_musique_hop.py`; `figure_data.md §H` |
| 17 | **drop_gold recovery**: cache recovers physically-removed answer evidence (text mirror of #3) | ✅ | ACC n=800/ds: tw drop_gold cov70 +3.4 pp (38:11), cov100 +2.1 pp (40:23); hp +1.6 pp (28:15); mq uniform recovery cell +3.2 pp (27:9); mq hop4 .101→.129; hero transcripts reconstruct removed 3-hop chains verbatim | `figure_data.md §H, §L` |
| 18 | **Mechanism = downstream-attention trace** (kept tokens that attended to dropped content during prefill carry its imprint) | ✅ CAUSAL ×2 families | gold-pos A/B, 3 datasets × 2 families, 5/5 cells: Qwen first +2.4 (mq n=2400, p=3e-4) / +3.6 (p=.002) / +4.1 (p=4e-4) vs last all null (mq −0.0 at n=2400); Mistral first +3.9 (p=.0013) / +5.8 (p<1e-4) vs last −0.2/−0.5 ns; observational split confirms | `figure_data.md §I` |
| 19 | Recovery reads out as **disambiguation + internal-knowledge framing, not hallucination** | ✅ | 91 recovery transcripts: dominant mode = tips correct same-named entity; recovered facts correct; over-anchoring is the separate loss population | `figure_data.md §L`; `mine_recovery.py` |
| 20 | HotpotQA mapped fresh-favored cell: gold-kept at cov50, −1.8 pp, via **distractor over-anchoring** at case level (cache amplifies kept topically-adjacent distractor over present gold); gone by cov70 | ✅ | cov50 gold-KEPT .748→.730 (n=408, 19:12); cov70 +0.5 pp; hp drop_gold recovery still positive (+1.6 pp). NB an interim n=304 partial run showed −5.2 pp — superseded by the full n=800 data; do not cite −5 pp | `figure_data.md §H, §L` |
| 21 | Scope boundary: sign = (need for dropped evidence) × (distractor adjacency); **not hop-count** | ✅ | mq/tw positive (shortcut-resistant), hp neutral w/ one fresh-favored cell (#20); hp recovers +4.1 pp when trace maximized (#18) | `figure_data.md §H, §I` |
| 22 | **Method A**: degeneration-gated adaptive coverage — cache-side signals predict errors; gate beats fixed-cov acc-vs-KV frontier; escalation needs no re-prefill under reuse (nested keep-sets) | ✅ POC (simulation on real sweeps) | P(wrong\|sig)=.80 vs .57 (mq c30); same acc w/ 9–22% less KV; tw gate dominates c50 (+1.5 pp, −9%) | `figure_data.md §K`; `gate_analysis.py` |
| 23 | **Method C**: position-aware keeping — at fixed budget keep LATER context | ✅ (scoped) | tw dd +3–4 pp (late +2.3/+4.6 vs early −0.9/+0.7, n=800); hp dd +2.1 at cov30 (late 46:30 p=.085, fading by cov50); mq ≈0. Keep-late never hurts; pays where the trace is strong. Fresh unchanged ⇒ cache-specific; discordant-pair collapse under keep-early | `figure_data.md §J` |
| 24 | Trace-aware (absorption-based) eviction as the axis importance-based methods miss | ⚠️ OUTLOOK | derived from #18; not yet benchmarked vs H2O/SnapKV — the sequel | README Act V.3 |
| 25 | **Cross-family replication**: coverage curve + causal A/B reproduce on Mistral-Small-24B (dense, non-Qwen) | ✅ | uniform 2wiki n=800: c50 +2.8 (p=.009), c70 +2.4 (p=.008), identity exact 0:0, gold-kept c10 cliff −3.5 pp; gold-pos first +3.9/+5.8 ≫ last ≈0 — effects larger than Qwen | `figure_data.md §M, §I`; `scripts/50_xfam_hop.py` |
| 26 | **Slot-0 is not special** — recovery tracks downstream mass, not primacy / attention-sink / lost-in-the-middle | ✅ | gold_pos=middle (seeded interior slot): slots j=1–2 (gp_after 7–8) = +4.9 pp in BOTH families ≈ gold-first; aggregate middle intermediate (+0.1…+1.5 ns); dose-response threshold-shaped | `figure_data.md §N` |
| 27 | Threshold shape = **bridge-routing**: the trace pays when it lands on the semantically-bound supporting paragraph, not arbitrary distractors | ✅ Mistral / ⚠️ Qwen directional | Mistral 2wiki middle: bridge-downstream +2.5 pp (18:7, p=.043) vs bridge-before −0.6 ns; Qwen same direction ns; hp flat at middle doses | `figure_data.md §N`; `analyze_mid.py` |
| 28 | **The trace stores CONTENT, not just retrieval priming** (counterfactual-gold probe) | ✅ existence; bandwidth ~1.4% | 2wiki n=461: cache reproduces a fabricated year that exists only in the deleted para's KV — 6:0 flips, exact p=.031 at max trace (dose: natural-pos 2:0); TRUE-year (priming) channel null (18:14 / 27:23); ctrl ceiling .931 + identity clean; partial fidelity (year right, day/month wrong) | `figure_data.md §O`; `scripts/51_counterfactual.py` |
| 29 | CF-probe sensitivity = (task ceiling) × (verbatim bandwidth); the MuSiQue null is explained, not contradictory | ✅ methodological | mq ctrl ceiling .420 (hop2 .57 → hop4 .22 = bridge failure before the gold para is ever read); .42 × 1.4% × 226 ≈ 1 item < detection | `figure_data.md §O` |

## Hard caveats to honor in the writing
- **Do not** claim an accuracy win for generic retrieval-RAG (#7).
- **Do not** claim "position-preserving beats compaction" (#10) or "we beat ReKV by
  keeping positions" — it's a tie (#11).
- Multi-hop instrument results are reported in **accuracy** (internal note: don't cite
  the instrument's NLL numbers in the post — generic metric caveats are not our story).
- The multi-hop story is now **resolved** — tell it as artifact-found → instrument-
  redesigned → mechanism-proven (#16→#18), not as an open problem. §D is historical.
- Method A is a **simulation POC** on real sweep data (say so); Method C is a real
  controlled run. B (#24) is outlook only — no benchmark claims.
- drop_gold absolute accuracies are intentionally low (evidence removed); the claim
  is the cache−fresh **relative** gap, never absolute performance.
- Cross-modal wording: "Cross-Modal Pattern Completion via Contextualized KV
  Imprinting" / "Modality-Absent Associative Recall" — and note the audio KV is
  *physically removed*, not masked (verified in `44_omni_coverage.py`
  DROP_AUDIO_AT_USE), so recovery comes purely from the video-token KV imprint.
- CF probe (#28): sell it as an **existence proof + bandwidth measurement**, not as
  "the cache stores everything" — net verbatim transmission ≈ 1.4%/readable item;
  the bulk of the accuracy benefit still flows through disambiguation (#19). Always
  pair the 6:0 with the ctrl ceiling (.931) and the priming-null.
- Middle ablation (#26/#27): the **aggregate** middle cells are ns — the claims live
  in the high-dose bin (slot-0-not-special) and the bridge stratification. Bridge-
  routing is significant on Mistral only; say "directional" for Qwen. MuSiQue dose
  bins are noise (n≈270/bin); per-bin item composition is shared across families
  (same seeded j), so bins are paired-valid but not independent replications.
- Mistral runs use the style-adapted instrument (`50_xfam_hop.py --style mistral`,
  BOS+`[INST]` prefix always kept); sanity = bit-exact cov100 NLL identity + EOS-
  clean gens. Don't compare absolute accuracies across families — compare gaps.

## Models / setups (for the methods paragraph)
- **Text (coverage curves):** Qwen3-30B-A3B-Instruct-2507 (+ Qwen3.5-27B thinking for
  the sink-dup diag), 48-layer full-attn MoE; LongBench 2wikimqa/hotpotqa/musique;
  chunk 256; KV splice at HF cache level.
- **Text (multi-hop instrument, §H–§L):** same A3B-Instruct; original MuSiQue
  ans-v1.0 dev (20 paras, hop 2/3/4), HotpotQA dev-distractor (10 paras), 2WikiMQA dev
  (10 paras); distractor single-pass, seeded uniform paragraph-level KV-compression;
  modes uniform / drop_gold / gold_pos{first,last} / keep_bias{early,late}; 3 arms
  fresh / origpos / compact; greedy gen + alias-match ACC (gens persisted), exact
  cov100 identity gate in every run; n=796–800 per accuracy cell family (NLL sweeps
  n=1500); gold_pos extended {first,last,middle} (middle = seeded interior slot,
  gp_slot/gp_after recorded); MuSiQue gold-pos cells extended to n=2400 (~full dev).
- **Cross-family (§M, §I-Mistral, §N):** Mistral-Small-24B-Instruct-2501 (40-layer
  dense, standard RoPE), `scripts/50_xfam_hop.py` `--style mistral` (family chat
  markers + extra-EOS; BOS+`[INST] ` doc-prefix tokens always kept, subsuming the
  sink; `fix_mistral_regex` tokenizer flag); same datasets/seeds/metric.
- **Counterfactual probe (§O):** `scripts/51_counterfactual.py` on A3B; answer-year
  swap inside the gold para before prebake (Δ∈{3,4,6,7,8,11,13} seeded per item);
  5-way eligibility filter (year in answer; in gold para only; not in question; no
  fabricated-year collision); metric = word-bounded year-token match for BOTH years
  per arm; cells {2Wiki,MuSiQue} × {uniform-ctrl, drop_gold, drop_gold+gold-first},
  cov100, eligible n = 461/226.
- **Video / cross-modal:** Qwen3-Omni-30B-A3B Thinker (48-layer full-attn, M-RoPE);
  EgoSchema-Subset + Video-MME; position-preserving M-RoPE splice; paired gold-answer
  NLL, Wilcoxon; cov100 identity gate; bf16 with fp32 control.
