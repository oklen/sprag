# Blog material — *The memory in a reused KV-cache*

Writing brief for the KV-cache-reuse blog. Audience: **senior ML researchers**
(academic-restrained tone). Spine = **we propose and validate one method
(position-preserving full-context KV reuse); the negative results refute the
competing explanations/approaches, not our method.** Every number here is
traceable in [`claims_ledger.md`](claims_ledger.md); plot-ready tables in
[`figure_data.md`](figure_data.md).

---

## 0. One sentence

> A KV-cache built over the **full** context is a compressed associative memory of
> it; reusing it on a kept subset (at original positions) beats recomputing over
> that subset, because the cache **remembers the context you dropped** — and this
> one mechanism holds across text long-docs, video frame-eviction, and cross-modal
> (audio→video) recall.

The field treats KV-cache reuse as a **systems/latency** problem (TTFT). The claim
here is that it has an unexamined **quality** dimension: reuse is a *memory*, and
that memory measurably helps (and sometimes hurts) accuracy.

## 1. The hook (open with this) — cross-modal associative recovery

Prebake a video clip **with its audio**. At use-time **delete the audio entirely**
and keep only the video tokens' KV. Ask an audio-dependent question.

- The reused cache (video KV that was baked *next to* the audio) **still answers**,
  even though the audio is physically gone at inference.
- Control: re-encode the same frames at the same positions but **without** audio in
  the prebake → the advantage vanishes to **exactly 0**.

Numbers (Qwen3-Omni-30B, Video-MME, n=597, paired gold-answer NLL, lower=better):

> At full visual coverage, audio-prebaked-cache − audio-free-recompute = **−0.070
> NLL (p<1e-4)**, while the audio-free vision baseline is **+0.000** (identity).
> The −0.07…−0.13 gap persists at every coverage and is invariant to how frames are
> evicted ⇒ a genuine **cross-modal trace in the video KV**, not a coverage artifact.
> Δacc +6.9 pp at cov20.

This is the wow: a prebaked cache holds *cross-modal associative memory*. It is the
differentiator vs ReKV (sliding-window, local) and MuKV (query-time compression).

## 2. Hero figure — the coverage curve (and its cliff)

One figure carries the whole thesis: **gap (cached − fresh) vs how much context you
keep**. Overlay text and video; annotate the low-coverage cliff. (Data:
`figure_data.md` §A, §B.)

The shape *is* the evidence:
- **Monotone in coverage** — the bonus is largest when recompute is most starved.
- **Converges to exactly 0 at full coverage** — the built-in identity gate; an
  artifact would not vanish exactly. (At cov100 nothing is dropped → nothing to
  remember.)
- **Video is monotone all the way down**; cov10 still beats fresh (ΔNLL −0.229;
  short clips **+18 pt accuracy**, .33→.51).
- **Text has a low-coverage CLIFF**: below a threshold the cache is *worse* than
  fresh (at c0, +0.5 NLL, million-scale PPL = decode degeneration). The bonus only
  appears once enough context is retained (crossover ~c25–c50).

### The mechanism: two competing forces, coverage is the knob
- **(+) memory bonus** — cached KV carry a trace of the dropped context; fresh
  recomputes over the starved subset only. Grows the *more* you drop.
- **(−) fragment/degeneration cost** — a cache over a *tiny* keep-set is a degenerate
  prefix; decode collapses. Worst at the extreme low end.

Low coverage on text: (−) wins → cliff. Mid: (+) wins → cache beats fresh. Full:
both → 0 → curves meet. Video never reaches the text-c0 starvation (cov10 still
keeps the sink + several temporal groups), so no cliff.

## 3. Why text "couldn't see it" but video could (the narrative pivot)

The honest origin story. The text **full-attention** splice experiment first looked
like a *null*: cached ≈ fresh. Reason: in those QA sets the answer sat **adjacent to
the query**, so a starved fresh recompute already had what it needed — the global
memory had nothing to contribute. Pivoting to **temporal video QA** (the answer
requires integrating across the whole timeline) gave the global memory something to
do — and the bonus appeared, significant and monotone. *Then* it reproduced back on
text once we measured accuracy on multi-fact questions. This pivot is worth telling:
it shows the mechanism's scope rather than overclaiming universality.

## 4. Scope boundary (state this prominently — it is the credibility)

The bonus exists **only** when the cache is built over a **unified** context and
inference uses a **subset** of it. Two regimes qualify: (a) multi-query over one long
document, (b) streaming video frame-eviction.

> In vanilla corpus-RAG — independently pre-baked chunks, each encoded in isolation —
> the information asymmetry is **zero**, so cached ≈ fresh and reuse is a pure latency
> play. **We never claim an accuracy win for generic retrieval-RAG.**

Pre-empting "isn't this just in-context information leakage?" by naming the boundary
*is* the rigor. (It is in-context memory — and we map exactly when it pays.)

## 5. What we ruled out (negatives as refutations of competing stories)

Frame each as: *the intuitive/competing explanation X fails; the real story is the
single memory bonus.* This section is the methodological-honesty backbone — and per
the chosen framing, every negative **strengthens** our method by killing an
alternative, not our own claim.

| Competing claim (intuitive / prior-art-implied) | Verdict | What actually holds |
|---|---|---|
| "Position-preserving reuse ≫ compaction" (an attack on InfLLM/ReKV repositioning) | **NULL** even under maximal M-RoPE shear (penalty +0.006, n.s.; does *not* concentrate on temporal/long clips) | Compaction is essentially **free** on this model; our win is the memory bonus, **not** keeping positions. We do **not** attack ReKV on repositioning. |
| "cached ≥ fresh is universal" | **False — conditional** | Text has a low-coverage cliff (cache worse), and it is **convention-independent** (origpos +0.519 ≈ compact +0.571 at c0) ⇒ the cliff is keep-set *starvation*, not a position-scramble. |
| "The cross-modal trace lives in deep layers" | **Wrong localization** | Recall is real (−0.100 NLL, 70% of records) but read out in **early–mid layers (~L8–24)**: first 8 layers realize 68% of the gap, plateau by L24. |
| "Sink-duplication harm = attention dilution (two sinks split the mass)" | **Falsified** | It is a **decode-trajectory** failure (an open `<think>` that never closes), a measurement bug we found and fixed — not attention sharing. |

The arc: kill the flashy sub-mechanisms; what survives — and is *reinforced* (the
M-RoPE rerun gave the largest accuracy lift, +18 pt on short video) — is the single
unified memory bonus.

## 6. Positioning vs prior work

- **TurboRAG** (isolated-chunk caching): TTFT-only, quality-neutral, no info-gap.
- **InfLLM / ReKV** (block KV retrieval + **compaction**): systems framing. Our
  faithful ReKV reimplementation **ties** position-preserving reuse (repositioning is
  near-free here, so it's a tie, not a win we claim over them).
- **MuKV** (dual-signal attn+FFT pruning): its advantage decomposes to **fine
  token-level selection** (+0.0152 of the gain) — the compaction it does is free
  (+0.0025) — and is **largely query-driven**: a query-free version helps only on
  long content (~⅓ of the gain) and **nothing** on short clips. ⇒ Query-aware
  compression isn't prebakeable; **position-preserving reuse is the right default**,
  and informed selection is an orthogonal, composable add-on (our `mukv@orig-pos` arm
  keeps both).
- The community (vLLM / LMCache) sees multimodal KV reuse only as a systems/TTFT
  problem. **The cross-modal recall niche is empty** — that's the opening.

## 7. Takeaways (for the reader)

- **Practitioners:** position-preserving reuse is the correct default; in streaming /
  multi-query regimes the accuracy bonus is *free*; don't over-evict (the cliff is real
  on text below ~a few-chunk keep-set).
- **Researchers:** reuse is a quality lever, not just latency; the cross-modal
  associative trace is an open, under-explored capability of prebaked caches.

## 8. Open problem (the honest coda — invites the community)

**Is the trace double-edged?** The bonus is significant where the task benefits —
hotpotqa (2-hop) **+9.5 pp accuracy, z=2.4**. On the multi-hop set (musique) the
cache is *negative* at low coverage (−6.8 pp) — consistent with a *partial reasoning
trace misleading* the model — **but it is not statistically significant** (z=−1.3),
and two 2-hop datasets disagree (2wikimqa is null), so the sign is a function of
**task structure, not hop-count**. Honest caveat to state: our text coverage is
**positional** (we keep the chunks physically before the answer), not **semantic**
(the supporting paragraphs), so this setup is not a clean multi-hop probe.

> Cleanly isolating *when the remembered trace helps vs misleads* — within one dataset,
> by reasoning-hop, with semantic (supporting-fact) coverage — is open. (We have a
> follow-up running on original MuSiQue with hop labels + semantic coverage; results
> will be appended.)

This is the kind of honest open question that ages well and gives the community a
foothold.

---

## Asset index
- `figure_data.md` — every plot-ready table (text/video coverage curves, cross-modal,
  by-dataset accuracy, layer-wise depth profile).
- `claims_ledger.md` — claim → status → evidence (n, stat, p) → script/data location.
- Source experiments in-repo: `experiments/cov_curve/`, `experiments/coverage_sinkdup/`,
  `experiments/omni_deepdive/`, `docs/OMNI_RESULTS.md`.
