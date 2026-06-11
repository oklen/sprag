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
hop-count. (Note: coverage here is *positional*, not semantic — the flaw the
redesigned instrument fixes; see the superseded-note above.)

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
shortcut-prone), 2WikiMQA (10 paras). All §H–§K numbers are **accuracy**
(alias-match on greedy generation; generations persisted per arm).

## §H — ACC matrix: uniform compression, 3 datasets (final, n=796–800 each)

`data/acc_{mq,hp,tw}_uniform.s*.json` · agg `agg_acc.py`. acc fresh / ours / compact.

**MuSiQue (n=796):**

| cov | ALL f/o/c | gold-KEPT f/o/c | gold-DROPPED f/o/c (recovery cell) |
|---:|---|---|---|
| 10  | .210/.205/.207 | .608/.519/.570 (cliff: cache worse) | .166/.170/.167 |
| 30  | .286/**.309**/**.318** (43:25) | .515/.515/.545 | .192/**.224**/**.224** (27:9) |
| 50  | .354/.367/.374 | .505/.530/.551 | .206/.206/.201 |
| 70  | .440/.441/.451 | .531/.538/.554 | .214/.201/.197 |
| 100 | .560/.560/.560 (identity, exact) | — | — |

**HotpotQA (n=800):**

| cov | ALL f/o/c | gold-KEPT f/o/c | gold-DROPPED f/o/c |
|---:|---|---|---|
| 10  | .429/.435/.435 | .716/.741/.753 | .396/.401/.399 |
| 30  | .535/.532/.536 | .733/.729/.744 | .441/.439/.437 |
| 50  | .611/.604/.610 | .748/**.730**/.745 (−1.8 pp, 19:12 — the one fresh-favored cell) | .469/.472/.469 |
| 70  | .667/.672/.675 | .733/.738/.738 | .511/.515/.523 |
| 100 | .766/.766/.766 (identity, exact) | — | — |

**2Wiki (n=800):**

| cov | ALL f/o/c | gold-KEPT f/o/c | gold-DROPPED f/o/c (recovery cell) |
|---:|---|---|---|
| 10  | .514/.525/.531 | .805/.782/.793 | .478/.494/.499 |
| 30  | .568/**.591**/.594 (38:19) | .747/.763/.780 | .490/**.517**/.513 (22:7) |
| 50  | .645/.656/.645 | .765/.767/.757 | .525/.545/.532 |
| 70  | .723/.735/.728 | .798/.816/.802 | .551/.551/.559 |
| 100 | .865/.865/.865 (identity, exact) | — | — |

**drop_gold ACC (answer-evidence physically removed; final, n=800 each):**

| cov | MuSiQue f/o/c | HotpotQA f/o/c | 2Wiki f/o/c |
|---:|---|---|---|
| 30  | .204/.200/.196 | .411/.420/.425 | .502/.512/.505 |
| 50  | .229/.231/.223 | .420/.429/.421 | .507/.526/.520 (31:16) |
| 70  | .219/.226/**.245** | .445/**.461**/.460 (28:15) | .515/**.549**/.542 (**+3.4 pp**, 38:11) |
| 100 | .255/.260/.265 | .502/.515/.516 (23:13) | .549/.570/.559 (40:23) |

(MuSiQue hop4 drop_gold c70: .101→.129, n=139. Note drop_gold cov100 is *not* an
identity gate — the gold paragraph is still removed.)

## §I — CAUSAL money table: gold-position A/B (drop_gold cov100, TWO model families)

`launch_goldpos.sh` / `launch_goldpos_4gpu.sh` / `launch_qext8.sh` (mq n=2400) /
`launch_xfam8.sh` (Mistral) · `data/gp_{hp,tw}_{first,last}.s*`,
`data/gp_mqext_{first,last}.s*`, `data/xf_{tw,hp}_gp{first,last}.s*`. **Identical
kept set; only the gold paragraph's prebake position varies.** gold-FIRST = every
kept paragraph attends to gold during prefill (max trace); gold-LAST = none do
(zero trace). p = exact McNemar on flips.

**Qwen3-30B-A3B-Instruct:**

| dataset | pos | n | acc fresh | acc ours | ours−fresh | flips | p |
|---|---|---:|---:|---:|---:|---|---|
| HotpotQA | first | 714 | .529 | .570 | **+.041** | 46:17 | 4e-4 |
| HotpotQA | last  | 800 | .512 | .510 | −.002 | 7:9 | ns |
| 2Wiki    | first | 800 | .551 | .588 | **+.036** | 55:26 | .002 |
| 2Wiki    | last  | 800 | .546 | .551 | +.005 | 14:10 | ns |
| MuSiQue  | first | **2400** | .280 | .304 | **+.024** | 148:91 | **3e-4** |
| MuSiQue  | last  | **2400** | .254 | .254 | −.000 | 45:46 | 1.0 |

(MuSiQue extended from n=800 to n=2400 ≈ full dev: the earlier marginal first-cell
(p=.10) is now decisive, and the earlier "last mildly positive +0.9pp" was noise —
last collapses to exactly zero. All three datasets: first significant, last null.)

**Mistral-Small-24B-Instruct-2501 (cross-family replication, n=800/cell):**

| dataset | pos | acc fresh | acc ours | ours−fresh | flips | p |
|---|---|---:|---:|---:|---|---|
| HotpotQA | first | .526 | .584 | **+.058** | 56:10 | <1e-4 |
| HotpotQA | last  | .532 | .527 | −.005 | 1:5 | ns |
| 2Wiki    | first | .529 | .568 | **+.039** | 60:29 | .0013 |
| 2Wiki    | last  | .535 | .532 | −.002 | 2:4 | ns |

Reading: first ≫ last, last ≈ 0, on **five of five dataset×family cells** ⇒ the
recovery **is** the downstream-attention trace, causally, and it is not a Qwen
quirk (Mistral effects are larger). Bonus signature: under gold-LAST the two arms
barely *disagree at all* (flips 1:5, 2:4) — remove the trace and the arms become
the same model. Observational confirmation (natural data, split by
#kept-paras-after-gold): 2Wiki cov100 +2.9 pp (after>0) vs −2.8 pp (after=0);
`pos_analysis.py`.

## §J — Method C: position-aware keeping (keep first-k vs last-k, n=800 each)

`--keep_bias {random,early,late}` · `data/kb_{tw,mq}_{early,late}.s*` · uniform,
cov30/50. Same budget; only *which* paragraphs are kept differs.

| dataset | cov | EARLY f/o (gap) | LATE f/o (gap) | diff-in-diff |
|---|---:|---|---|---:|
| 2Wiki | 30 | .573/.564 (−0.9 pp, 6:13) | .576/.599 (**+2.3 pp**, 41:23, p=.03) | +3.2 pp |
| 2Wiki | 50 | .630/.637 (+0.7 pp, 12:6) | .626/.672 (**+4.6 pp**, 59:22, p<1e-4) | +3.9 pp |
| HotpotQA | 30 | .491/.490 (−0.1 pp, 8:9) | .511/.531 (**+2.0 pp**, 46:30, p=.085) | +2.1 pp |
| HotpotQA | 50 | .559/.554 (−0.5 pp, 10:14) | .609/.613 (+0.4 pp, 29:26) | +0.9 pp |
| MuSiQue | 30 | .278/.287 (+0.9 pp) | .299/.305 (+0.6 pp) | ~0 |
| MuSiQue | 50 | .350/.352 (+0.2 pp) | .365/.375 (+1.0 pp) | +0.8 pp |

Scoreboard (3 datasets): 2Wiki strong (+3–4 pp dd), HotpotQA weak-positive (+2.1 dd
at cov30, fading by cov50), MuSiQue ≈ 0. Keep-late **never hurts** (early arm is the
one that goes negative); it pays where the trace is strong. `data/kb_hp_*.s0.json`.

Two annotations for the figure: (1) fresh is ~unchanged early vs late ⇒ clean
diff-in-diff, the benefit is cache-specific; (2) discordant pairs (items where
ours≠fresh) collapse under keep-early and blow up under keep-late — 2Wiki: 19/18
of 800 (early) vs 64/81 (late) at cov30/50; MuSiQue: 30/40 vs 77/76 — the trace in
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

**Australia conscription 1964 (3-hop; gold "1964"; quote is the c50 cell):**
- *fresh*: "The question contains a mix-up in details…" (hedges)
- *cached (origpos)*: "…*Grievous Bodily Harm* (1990) was released in Australia.
  …**Australia reintroduced conscription for the Vietnam War in 1964**, under the
  National Service Act 1964."
- *cached (compact)* errs to UK here — origpos-only win on this item. (At c70 the
  cached arm drifts to the 1965 call-up year — quote the c50 cell only.)

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

## §M — Cross-family replication: Mistral-Small-24B coverage curve (2Wiki, n=800)

`scripts/50_xfam_hop.py` (`--style mistral`: family chat markers, BOS+`[INST] `
doc-prefix always kept, `fix_mistral_regex`) · `data/xf_tw_uniform.s{0,1}.json` ·
launcher `launch_xfam8.sh`. Same instrument as §H, second model family.

| cov | ALL f/o/c | gap (flips, p) | gold-KEPT f→o | gold-DROPPED f→o |
|---:|---|---|---|---|
| 10  | .514/.522/.516 | +0.9 pp (27:20, ns) | .805→.770 (cliff) | .478→.492 |
| 30  | .564/.581/.568 | +1.8 pp (38:24, p=.098) | .772→.780 | .474→.496 |
| 50  | .610/.637/.639 | **+2.8 pp** (44:22, p=.009) | .723→.767 | .497→.507 |
| 70  | .698/.721/.720 | **+2.4 pp** (33:14, p=.008) | .778→.798 | .514→.547 |
| 100 | .850/.850/.850 | 0 (0:0) **identity exact** | — | — |

Every qualitative feature of the Qwen curve reproduces: cache ≥ fresh at all
coverages, mid-coverage peak, exact identity gate, gold-kept low-coverage cliff
(−3.5 pp at cov10), positive recovery cells throughout (up to +3.3 pp at cov70).
Sanity gate for the port: cov100 NLL bit-identical across arms; EOS-clean gens.

## §N — Position ablation: gold_pos=middle (primacy / lost-in-the-middle control)

`--gold_pos middle` = seeded random interior slot j∈[1, n−2] (≥1 kept para
downstream); `meta.gp_slot`/`gp_after` recorded per item. drop_gold cov100, n=800
per cell, both families. `data/{qm,xf}_{tw,hp}_gpmid.s*`, `data/qm_mq_gpmid.s*` ·
launchers `launch_xfam_mid.sh`, `launch_qmid8.sh` · analysis `analyze_mid.py`.

**Aggregate (middle sits between first and last):**

| cell | first | middle | last |
|---|---:|---:|---:|
| Qwen 2Wiki | +3.6 | +0.1 (24:23, ns) | +0.5 |
| Qwen HotpotQA | +4.1 | +0.5 (22:18, ns) | −0.2 |
| Qwen MuSiQue | +2.4 | +1.5 (30:18, p=.11) | −0.0 |
| Mistral 2Wiki | +3.9 | +1.1 (24:15, ns) | −0.2 |
| Mistral HotpotQA | +5.8 | +1.3 (22:12, ns) | −0.5 |

**Dose-response by gp_after (#kept paras downstream of gold), 10-para datasets:**

| gp_after bin | Qwen 2Wiki | Mistral 2Wiki | Qwen HP | Mistral HP |
|---|---:|---:|---:|---:|
| 1–3 (n=318) | −0.6 | +0.0 | −0.3 | +0.6 |
| 4–6 (n=300) | −2.0 | +0.0 | +0.0 | +1.7 |
| 7–8 (n=182) | **+4.9** | **+4.9** | **+2.7** | +1.6 |

Two readings, both load-bearing:
1. **Slot 0 is not special**: slots j=1–2 (bin 7–8) recover as much as gold-FIRST
   (+4.9 ≈ +3.6/+3.9) in both families ⇒ the primacy/attention-sink/L-i-t-M
   alternative is dead. The predictor is downstream mass, not the slot.
2. The response is **threshold-shaped**, not linear: most of the recovery needs
   (nearly) the whole kept context downstream.

**Why threshold? Bridge-routing refinement** (re-derive j from rid; stratify by
whether the OTHER supporting paragraph sits downstream of gold):

| middle run | bridge DOWNSTREAM | bridge all BEFORE |
|---|---|---|
| Mistral 2Wiki | **+2.5 pp** (18:7, p=.043) | −0.6 pp (6:8, ns) |
| Qwen 2Wiki | +0.4 pp (16:14, ns) | −0.3 pp (8:9, ns) |
| HotpotQA (both) | flat (+0.3…+1.2 ns) | flat |

The trace is useful when it lands on the **semantically bound** (supporting)
paragraph, not on arbitrary distractors — significant on Mistral 2Wiki, directional
on Qwen (whose threshold is steeper; its recovery needs near-total downstream
mass). Consistent with Method C (§J): keep-late retains the tokens downstream of
everything. Caveats: j is seeded per item (bins share item composition across
families — per-bin paired gaps are valid, cross-bin composition is not independent);
MuSiQue 20-para dose bins are noise-dominated at n≈270/bin.

## §O — Counterfactual-gold probe: content storage vs retrieval priming

`scripts/51_counterfactual.py` (49 + `_apply_cf`: swap the answer YEAR inside the
gold para before prebake; deterministic Δ∈{3,4,6,7,8,11,13}; eligibility = year in
answer & only in gold para & not in question & fabricated year collision-free;
metric = word-bounded year-token match, both years scored per arm, gens persisted)
· `data/cf_{tw,mq}_{ctrl,dgorig,dgfirst}.s*` · launcher `launch_cf8.sh` · eligible
n: 2Wiki 461, MuSiQue 226 (`cf_count.py`).

**2Wiki (n=461):**

| cell | FAB year fresh→cache | TRUE year fresh→cache |
|---|---|---|
| ctrl (uniform cov100, CF visible) | .931→.931 (0:0, identity) | .004→.002 |
| drop_gold, gold at natural pos | .000→.004 (2:0, ns) | .221→.230 (+0.9, 18:14 ns) |
| drop_gold, gold-FIRST (max trace) | .002→.015 (**6:0, exact p=.031**) | .217→.226 (+0.9, 27:23 ns) |

Verdict: (1) **content storage exists** — at max trace the cache reproduces a
fabricated year that exists nowhere except in the deleted paragraph's KV imprint
(6:0 one-sided; dose-consistent: natural 2:0 → first 6:0); (2) the **priming
channel is null** — TRUE-year reproduction is statistically equal between arms;
(3) **bandwidth is low**: net verbatim transmission ≈ 1.4% per readable item
(year-token lower bound), and the cache often gets the year right but the
day/month wrong ("April 15, 1946" vs in-doc "November 10, 1946") — a
partial-fidelity imprint. Control facts: ceiling .931 = the model trusts
in-context counterfactuals; ctrl arms bit-identical (identity gate holds for the
CF pipeline).

Hero transcripts (`cf_tw_dgfirst`, verbatim): fresh "the provided text does not
include Michael Schultz's date of birth" / cache "His birthday is **April 15,
1946**" (fab year, removed para); fresh wrong-guess "died on 5 December 2014" /
cache "died in **1998**" (fab); one cache case even claims "This is supported by
the text provided" — confabulated attribution of a real KV trace.

**MuSiQue (n=226): probe null — and the null is explained by the control ceiling.**
ctrl CF-echo ceiling is only .420 (by hop: 2-hop .57, 3-hop .47, 4-hop .22 — the
model fails the *bridge* before ever reading the gold para). Probe sensitivity =
ceiling × verbatim bandwidth ≈ .42 × 1.4% × 226 ≈ 1 item — below detection.
Consistent with, not contradicting, the 2Wiki positive (ceiling .93). Report the
formula; gens are normal (coherent reasoning, EOS-clean) — the instrument is fine,
the task ceiling binds.
