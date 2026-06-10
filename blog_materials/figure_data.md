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

## §D — [SUPERSEDED — historical only] LongBench by-dataset accuracy gap

> **Do not plot as a result.** This table came from the LongBench *positional*-coverage
> setup and predates the redesigned multi-hop instrument (uniform compression +
> identity gate + ACC, §H). Its open question is now **answered** (§H–§J). Keep only
> if telling the origin story of how the artifact was found.

`experiments/coverage_sinkdup` / `scripts/34_a3b_diag.py` + `data/a3b_cov_fix.s*`.
z = mean/SEM of the paired per-item acc difference.

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

---

# New sections (2026-06-10): the redesigned multi-hop instrument and its results

Instrument: `scripts/49_musique_hop.py` — distractor single-pass setting, **seeded
uniform KV-compression over all paragraphs** (no privileged answer-keep), 3 arms
(fresh / ours=origpos / compact), batched generation with persisted gens, **exact
cov100 identity gate** in every run. Model Qwen3-30B-A3B-Instruct-2507. Datasets:
MuSiQue (20 paras, hop 2/3/4, shortcut-resistant), HotpotQA (10 paras, 2-hop,
shortcut-prone), 2WikiMQA (10 paras). **Accuracy (alias-match on greedy gen) is the
metric of record**; NLL was found calibration-only here (see ledger #20).

## §H — ACC matrix: uniform compression, 3 datasets (n≈650–800 each)

`data/acc_{mq,hp,tw}_uniform.s*.json` · agg `agg_acc.py`. acc fresh / ours / compact.

**MuSiQue (n=650):**

| cov | ALL f/o/c | gold-KEPT f/o/c | gold-DROPPED f/o/c (recovery cell) |
|---:|---|---|---|
| 10  | .220/.217/.215 | .656/.562/.625 (cliff: cache worse) | .172/.179/.171 |
| 30  | .297/.314/.323 | .537/.526/.553 | .198/**.226**/**.228** (21:8) |
| 50  | .357/.374/.385 | .509/.544/.576 | .213/.213/.204 |
| 70  | .435/.437/.446 | .527/.536/.549 | .212/.196/.196 |
| 100 | .560/.560/.560 (identity, exact) | — | — |

**HotpotQA uniform (n=304, by gold_kept — the one mapped negative cell):**

| cov | gold-KEPT f/o/c | gold-DROPPED f/o/c |
|---:|---|---|
| 50 | .766/**.714**/.740 (cache −5.2 pp, 2:10) | .460/.480/.473 |
| 70 | .752/.757/.752 (tied) | .468/.479/— |

**2Wiki**: cache > fresh +1–3.4 pp across covs (see `acc_tw_*`; also §J random-keep
baselines: cov30 .576→.599, cov50 .626→.672 under keep-late).

**drop_gold ACC (answer-evidence physically removed; MuSiQue n=766):**

| cov | f / o / c | flips o>f:f>o |
|---:|---|---|
| 30  | .205/.204/.201 | 23:24 |
| 50  | .232/.235/.227 | 27:25 |
| 70  | .222/.230/**.247** | 28:22 |
| 100 | .258/.262/.268 | 20:17 |

(2Wiki drop_gold recovery, earlier run n=400: cov50 .215→.230/.233 (13:7), cov70
.205→.233/.230 (15:4); hop4 cov70 .055→.096, 3:0.)

## §I — CAUSAL money table: gold-position A/B (drop_gold cov100, n=714–800/cell)

`launch_goldpos.sh` / `launch_goldpos_4gpu.sh` · `data/gp_{hp,tw,mq}_{first,last}.s*`
· agg `agg_goldpos.py`. **Identical kept set; only the gold paragraph's prebake
position varies.** gold-FIRST = every kept paragraph attends to gold during prefill
(max trace); gold-LAST = none do (zero trace).

| dataset | pos | n | acc fresh | acc ours | ours−fresh | flips o>f:f>o |
|---|---|---:|---:|---:|---:|---|
| HotpotQA | first | 714 | .529 | .570 | **+.041** | 46:17 |
| HotpotQA | last  | 800 | .512 | .510 | −.002 | 7:9 |
| 2Wiki    | first | 800 | .551 | .588 | **+.036** | 55:26 |
| 2Wiki    | last  | 800 | .546 | .551 | +.005 | 14:10 |
| MuSiQue  | first | 800 | .284 | .305 | **+.021** | 55:38 |
| MuSiQue  | last  | 800 | .249 | .258 | +.009 | 19:12 |

Reading: first ≫ last on all three datasets ⇒ the recovery **is** the downstream-
attention trace, causally. MuSiQue's last-cell stays mildly positive (deep-hop tasks
benefit from cache even at natural positions); the ordering is what's universal.
Observational confirmation (natural data, split by #kept-paras-after-gold): 2Wiki
cov100 +2.9 pp (after>0) vs −2.8 pp (after=0); `pos_analysis.py`.

## §J — Method C: position-aware keeping (keep first-k vs last-k, n=800 each)

`--keep_bias {random,early,late}` · `data/kb_{tw,mq}_{early,late}.s*` · uniform,
cov30/50. Same budget; only *which* paragraphs are kept differs.

| dataset | cov | EARLY f/o (gap) | LATE f/o (gap) | diff-in-diff |
|---|---:|---|---|---:|
| 2Wiki | 30 | .573/.564 (−0.9 pp, 6:13) | .576/.599 (**+2.3 pp**, 41:23) | +3.2 pp |
| 2Wiki | 50 | .630/.637 (+0.7 pp, 12:6) | .626/.672 (**+4.6 pp**, 59:22) | +3.9 pp |
| MuSiQue | 30 | .278/.287 (+0.9 pp) | .299/.305 (+0.6 pp) | ~0 |
| MuSiQue | 50 | .350/.352 (+0.2 pp) | .365/.375 (+1.0 pp) | +0.8 pp |

Two annotations for the figure: (1) fresh is ~unchanged early vs late ⇒ clean
diff-in-diff, the benefit is cache-specific; (2) discordant pairs collapse under
keep-early (6–21 of 800) and blow up under keep-late (41–81) — the trace in
late-kept tokens *is* the difference between arms. compact gains under late too
(tw c30 .606) ⇒ trace survives re-rotation.

## §K — Method A: degeneration-gated adaptive coverage (simulation on §H data)

`gate_analysis.py` (local tmp). Cache-side runtime signals on the generated text:
abstention/hedge regex, repeated-4-gram fraction >0.3, no-EOS truncation.

**Signal predicts errors** — P(wrong | signal) vs P(wrong | none):

| dataset | c30 | c50 | c70 |
|---|---|---|---|
| MuSiQue  | .80 / .57 | .77 / .49 | .75 / .41 |
| HotpotQA | .70 / .40 | .65 / .34 | .53 / .30 |
| 2Wiki    | .44 / .39 | .40 / .32 | .37 / .23 |

**Gate policy** (answer at low cov; escalate while signal fires). avg_ntok = final
kept tokens (KV budget); total = incl. retried decodes (~2×; escalation under cache
reuse costs **no re-prefill** — nested keep-sets ⇒ fetch more rows of the stored
cache).

| dataset | policy | acc | avg KV toks | nearest fixed-cov comparison |
|---|---|---:|---:|---|
| MuSiQue  | gate@30 | .436 | 1505 | ≈ c70 acc (.441) with **−14% KV** |
| HotpotQA | gate@30 | .608 | 547  | ≈ c50 acc (.604) with **−22% KV** |
| 2Wiki    | gate@30 | .671 | 451  | **dominates** c50 (.656 @ 497): +1.5 pp AND −9% KV |
| 2Wiki    | oracle  | .922 | 408  | per-sample best cov; > c100 acc (.865 @ 979) at 42% budget |

Caveats to print: trunc signal partly an artifact of max_new=200; oracle is an upper
bound (any-cov-correct selection), not attainable.

## §L — Hero transcripts (verbatim, `data/dropgold_hero.json`, drop_gold c50)

Answer-evidence paragraph **physically removed**; same kept set for both arms.

**Vatican City (3-hop; gold "11 February 1929"):**
- *fresh*: "The question appears to be based on a misunderstanding… there is no
  direct connection…" (gives up)
- *cached*: "The author of *Princeps Pastorum* is **Pope John XXIII**, who died in
  Vatican City. Vatican City became an independent country on **11 February 1929**,
  when the Lateran Treaty was signed…" (full 3-hop chain, exact removed date)

**Warner Music Group (3-hop; gold "Warner Music Group"):**
- *fresh*: stops at "**Warner Records** owns the record label."
- *cached*: identical prefix, then "…owned by Warner Records, **a subsidiary of
  Warner Music Group**." (adds exactly the removed parent-company hop)

**Australia conscription 1964 (3-hop; gold "1964"):**
- *fresh*: "The question contains a mix-up in details…" (hedges)
- *cached (origpos)*: "…*Grievous Bodily Harm* was released in Australia. Australia
  reintroduced conscription for the Vietnam War era… **1964**."
- *cached (compact)* errs to UK here — origpos-only win on this item.

**Loss-side example (HotpotQA distractor over-anchoring, `data/hp_fail_dump.json`):**
The Hard Way (gold **Mos Def**, his paragraph removed, distractor "The Hard Way
(1991 film)" kept): *fresh* bridges parametrically to "Mos Def" ✓; *cached* answers
"**Michael J. Fox & James Woods**" — the kept distractor's literal cast ✗.

**Reasoning-mode examples (2Wiki recovery, `mine_recovery.py`, 91 cases):**
- Disambiguation: "Beatrice of England" — fresh: "…Isabella of France — wait, no,
  that's a mix-up…" picks the wrong Beatrice; cached picks Beatrice (1242–1275) →
  correct grandmother **Isabella of Angoulême**.
- Internal-knowledge framing: Ri Sol-ju's father-in-law — fresh hedges
  ("undisclosed"); cached states "Kim Jong-un's father, **Kim Jong-il**" as world
  knowledge (it cannot cite the removed paragraph).
