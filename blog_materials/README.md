# Blog material — *The memory in a reused KV-cache*

Writing brief for the KV-cache-reuse blog. Audience: **senior ML researchers**
(academic-restrained tone). Spine = **a reused KV-cache is an information-asymmetric
object: it remembers context you deleted. We show the phenomenon, prove the mechanism
causally, map its boundary honestly, and turn the explanation into two deployable
methods.** Every number is traceable in [`claims_ledger.md`](claims_ledger.md);
plot-ready tables in [`figure_data.md`](figure_data.md).

Angle (decided): **mechanism / interpretability first, practical implications last.**
Text and cross-modal are one merged story — the same mechanism in two modalities.

---

## 0. One sentence

> A KV-cache built over the **full** context is a compressed associative memory of
> it: reusing it on a kept subset beats recomputing over that subset, because the
> cache **remembers the context you dropped** — we trace this to **downstream
> attention** (kept tokens that attended to the dropped content during prefill carry
> its imprint), prove it with a controlled position experiment on three datasets and
> two modalities, and derive two practical compression policies directly from it.

The field treats KV-cache reuse as a **systems/latency** problem (TTFT). The claim
here is that it has an unexamined **quality** dimension: reuse is a *memory*, and
that memory measurably helps (and, in mapped conditions, slightly hurts) accuracy.

## Act I — The phenomenon: the coverage curve (open with this)

One figure carries the thesis: **gap (cached − fresh) vs how much context you keep**.
Overlay text and video; annotate the low-coverage cliff. (Data: `figure_data.md` §A, §B.)

The shape *is* the evidence:
- **Converges to exactly 0 at full coverage** — the built-in identity gate; an
  artifact would not vanish exactly. (At cov100 nothing is dropped → nothing to
  remember.) Every experiment in this post carries this gate.
- **The bonus grows the more you drop** — largest where recompute is most starved.
- **Video is monotone all the way down** (cov10 still beats fresh, ΔNLL −0.229;
  short clips **+18 pt accuracy**, .33→.51).
- **Text has a low-coverage CLIFF**: below a threshold the cache is *worse* than
  fresh (at c0, +0.5 NLL, million-scale PPL = decode degeneration). The bonus
  appears once enough context is retained (crossover ~c25–c50).

### Two competing forces; coverage is the knob
- **(+) memory bonus** — cached KV carry a trace of the dropped context; fresh
  recomputes over the starved subset only.
- **(−) degeneration cost** — a cache over a *tiny* keep-set is a degenerate prefix;
  decode collapses. Worst at the extreme low end; convention-independent (it is
  keep-set starvation, **not** position-scramble — origpos cliffs identically).

Low coverage on text: (−) wins → cliff. Mid: (+) wins → cache beats fresh. Full:
both → 0. Video never reaches text-c0 starvation, so no cliff.

## Act II — The same memory across modalities (cross-modal recovery)

Prebake a video clip **with its audio**. At use-time **physically delete the audio**
and keep only the video tokens' KV. Ask an audio-dependent question.

- The reused cache (video KV baked *next to* the audio) **still answers**.
- Control: re-encode the same frames at the same positions **without** audio in the
  prebake → the advantage vanishes to **exactly 0**.

Numbers (Qwen3-Omni-30B, Video-MME, n=597, paired gold-answer NLL): at full visual
coverage, audio-prebaked − audio-free = **−0.070 NLL (p<1e-4)** while the audio-free
baseline is +0.000; gap invariant to eviction mode; **Δacc +6.9 pp at cov20**.
(Data: `figure_data.md` §C.)

This generalizes Act I beyond text — the memory is not a text quirk; it transfers
across modalities inside one unified cache.

## Act III — The mechanism, nailed (the climax)

### III.1 Text mirror of the cross-modal test: `drop_gold`
Multi-hop QA (MuSiQue / HotpotQA / 2WikiMQA), distractor setting, seeded uniform
KV-compression with the cov100 identity gate. Now **always physically remove the
answer-evidence paragraph**; coverage acts on the rest. Fresh recompute never sees
the evidence. The reused cache recovers it: ACC cache > fresh in the recovery cells
— gold-dropped stratum of the uniform sweep: MuSiQue **+3.2 pp** at cov30 (27:9
flips), 2Wiki +2.7 pp (22:7); dedicated drop_gold mode (n=800/dataset): 2Wiki cov70
**+3.4 pp** (38:11), HotpotQA +1.6 pp (28:15), MuSiQue hop4 .101→.129. Hero
transcripts (`figure_data.md` §L): fresh "I can't determine this" / cache
reconstructs a 3-hop chain ending in the exact removed date.

### III.2 Where does the recovered information live? Downstream attention.
Hypothesis: during full prefill, kept paragraphs positioned **after** the gold
paragraph attended to it (causal mask), so their K/V absorbed gold-derived
information; deleting gold leaves that imprint in the surviving KV. Fresh recompute
never had it.

- **Observational split**: recovery exists only when ≥1 kept paragraph sits after
  gold (2Wiki cov100: +2.9 pp with downstream attenders vs −2.8 pp with none).
- **Controlled causal A/B** (the money table, `figure_data.md` §I): same kept set,
  only the gold paragraph's *prebake position* varies.

| dataset | gold-FIRST (all kept paras attend it) | gold-LAST (none do) |
|---|---|---|
| HotpotQA | **+4.1 pp** (46:17) | −0.2 pp (7:9) |
| 2Wiki    | **+3.6 pp** (55:26) | +0.5 pp (14:10) |
| MuSiQue  | **+2.1 pp** (55:38) | +0.9 pp (19:12) |

Maximize downstream attenders → recovery is largest; eliminate them → it collapses.
Note HotpotQA: *neutral* in natural data, **+4.1 pp when the trace is maximized** —
the mechanism is universal; natural gold position just modulates it.

### III.3 What does recovery look like in the model's reasoning?
Reading 91 recovery transcripts (gens persisted): the dominant mode is
**disambiguation** — the trace tips the model to the correct same-named entity where
fresh wavers ("…wait, no, that's a mix-up…" → wrong Beatrice; cache picks the right
one). The model presents the recovered fact as its **own world knowledge** (it cannot
cite text that isn't there). In gain cases it is **not hallucination** — the
recovered facts are correct; hallucination/over-anchoring is the separate *loss*
population (Act IV). Characterization: the trace is a **soft prior / disambiguator**,
lossy (+2–4 pp, never verbatim injection).

## Act IV — Scope boundary (state prominently; it is the credibility)

Honest cross-dataset accuracy (uniform compression, all numbers are answer accuracy;
`figure_data.md` §H, n=796–800/dataset):
- **2Wiki**: cache > fresh at every coverage (+1.1–2.3 pp ALL; drop_gold recovery up
  to **+3.4 pp**, 38:11) — the clean win.
- **MuSiQue**: cache ≥ fresh (+1.3–2.3 pp at cov30–50; recovery cell +3.2 pp, 27:9).
- **HotpotQA**: ≈ neutral; **one mapped fresh-favored cell**: gold-kept at cov50,
  −1.8 pp (19:12) — *distractor over-anchoring* at case level (cache amplifies a
  kept, topically-adjacent distractor over the present gold; e.g. answers the kept
  1991 film's cast instead of bridging to the removed entity). Gone by cov70; and
  drop_gold recovery is positive even here (+1.6 pp at cov70).

Unifying boundary statement: the cached trace **amplifies whatever was salient in
the full context**. It helps when the dropped content is the needed signal and the
task can't be shortcut (deep-hop, shortcut-resistant: MuSiQue, 2Wiki). It costs a few
points when kept distractors are strong, coverage is low, and the task is
shortcut-prone (HotpotQA 2-hop). Sign = (need for dropped evidence) × (distractor
adjacency) — **not** hop-count (the old hop-count story is dead; see ledger #15).

And the standing boundary from before: the asymmetry requires a **unified-context
prebake + subset inference** (multi-query long-doc; streaming eviction). In
vanilla corpus-RAG with independently-baked chunks the asymmetry is zero — **no
accuracy claim for generic retrieval-RAG.**

## Act V — From explanation to method (practical implications)

The mechanism directly yields compression policy. Two methods made solid here; one
saved as the sequel.

### V.1 Degeneration-gated adaptive coverage (`figure_data.md` §K)
The cliff is detectable **from the cache side alone** (no fresh control): abstention
phrasing, n-gram repetition, no-EOS truncation predict wrong answers everywhere
(P(wrong|signal)=.80 vs .57 on MuSiQue c30). Policy: answer at low coverage,
escalate only when the signal fires. Result: a better acc-vs-KV frontier on all
three datasets (same accuracy with **9–22% less KV**; on 2Wiki it strictly dominates
fixed coverage). Oracle headroom is large (2Wiki .922 @ 42% of the full budget).

**The synergy that makes it a KV-reuse method**: keep-sets are nested and the stored
full-document cache already contains every row — escalation = **fetch more rows +
re-decode, no re-prefill**. Under fresh compute, escalation costs a full re-prefill.
Adaptive coverage is uniquely cheap in exactly the regime this post is about.

### V.2 Position-aware keeping (`figure_data.md` §J)
Causal corollary: later tokens carry the trace of earlier dropped content; earlier
tokens carry nothing of later content. So at a fixed budget, **prefer keeping later
context**. Controlled test (same budget, keep first-k vs last-k): 2Wiki cache−fresh
gap = −0.9/+0.7 pp (early) vs **+2.3/+4.6 pp (late)** at cov30/50 — a +3–4 pp
diff-in-diff, with fresh itself unchanged (.573 vs .576) ⇒ the benefit is
cache-specific, not "late content is better". Bonus mechanistic signature: under
keep-early, cache and fresh disagree on almost nothing (2Wiki: 19/18 items of 800);
under keep-late they diverge (64/81) — **the trace in late-kept tokens *is* the
difference between the two arms.** Zero-cost recipe; one sentence to implement.

### V.3 Outlook (the sequel): trace-aware eviction
Importance-based eviction (H2O/SnapKV) keeps what's attended-to. The mechanism adds
an orthogonal axis: a chunk whose information has already been **absorbed** by
surviving downstream tokens is safe to evict *regardless of its importance* — its
trace remains. An absorption/redundancy criterion, computable from prebake attention.
Stated as the open direction; benchmarked in the next post.

## What we ruled out (negatives as refutations of competing stories)

| Competing claim | Verdict | What actually holds |
|---|---|---|
| "Position-preserving reuse ≫ compaction" (attack on InfLLM/ReKV repositioning) | **NULL** (penalty +0.006, n.s., even under maximal M-RoPE shear) | Compaction ≈ free on this model; the win is the memory bonus, not positions. |
| "cached ≥ fresh is universal" | **False — conditional** | Text low-coverage cliff; convention-independent (starvation, not scramble). |
| "Cache hurts multi-hop reasoning" (our own early Path B result) | **Instrument artifact** | Old chain-mode kept the answer para always → fresh became an extractive oracle; alias-match penalized co-referent answers. Redesigned instrument (uniform + identity gate) erases it. |
| "The penalty/benefit tracks hop-count" | **Dead** | Sign tracks (need for dropped evidence) × (distractor adjacency); HotpotQA recovers +4.1 pp when the trace is maximized. |
| "The cross-modal trace lives in deep layers" | **Wrong localization** | Early–mid readout (~L8–24): first 8 layers = 68% of gap, plateau by L24. |
| "Sink-duplication harm = attention dilution" | **Falsified** | Decode-trajectory failure (unclosed `<think>`), a measurement bug we found and fixed. |

The arc: kill the flashy sub-mechanisms and our own early misreadings; what survives
is one mechanism — the downstream-attention trace — now causally proven.

## Positioning vs prior work

- **TurboRAG** (isolated-chunk caching): TTFT-only, quality-neutral, no info-gap.
- **InfLLM / ReKV** (block KV retrieval + compaction): systems framing; our faithful
  ReKV reimplementation **ties** position-preserving reuse (we claim no win there).
- **MuKV**: advantage decomposes to fine token-level selection, largely query-driven;
  not prebakeable. Position-preserving reuse is the right default; informed selection
  is an orthogonal, composable add-on.
- **H2O / SnapKV** (importance-based eviction): the natural foil for V.3 — absorption
  is the axis importance does not measure.
- The community (vLLM / LMCache) sees multimodal KV reuse as systems/TTFT only.
  **The cross-modal + mechanism + policy niche is empty** — that's the opening.

## Takeaways

- **Practitioners:** position-preserving reuse is the correct default; in
  streaming / multi-query regimes the accuracy bonus is free; don't over-evict (the
  text cliff is real); at a fixed budget keep *later* context; gate coverage on
  cache-side degeneration signals (escalation costs no re-prefill).
- **Researchers:** reuse is a quality lever, not just latency; the downstream-
  attention trace is causally established, lossy, and reads out as disambiguation;
  absorption-aware eviction is open.

## Origin-story note (use in Act I or a sidebar)

The text full-attention splice first looked like a null (answer adjacent to query →
fresh already had what it needed). Temporal video QA gave the global memory something
to do; the bonus appeared, then reproduced back on text once coverage was semantic
and the metric was accuracy. Tell this pivot — it shows scope discovery rather than
overclaiming, and it explains *why* the field missed the quality dimension.

---

## Asset index
- `figure_data.md` — every plot-ready table (coverage curves §A/§B, cross-modal §C,
  ACC matrix §H, causal A/B §I, keep-bias §J, gate §K, hero transcripts §L).
- `claims_ledger.md` — claim → status → evidence (n, stat) → script/data location.
- Source experiments in-repo: `experiments/cov_curve/`, `experiments/coverage_sinkdup/`,
  `experiments/omni_deepdive/`, `docs/OMNI_RESULTS.md`; multi-hop instrument
  `scripts/49_musique_hop.py` (uniform / drop_gold / gold_pos / keep_bias modes,
  batched ACC with persisted generations).
