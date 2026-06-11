# Your KV-Cache Remembers the Context You Deleted

*Training-free KV reuse is already a quality lever — the knob is coverage, the
mechanism is a downstream-attention trace, and we prove it causally on three
datasets and two modalities.*

---

## TL;DR

- A KV-cache built over the **full** context, then reused on a kept **subset**, is
  not equivalent to recomputing over that subset. It is an information-asymmetric
  object: it carries a measurable imprint of the context you deleted.
- This needs **no training**. Plain, position-preserving reuse — no learned
  compressor, no auxiliary model, no fine-tuning — beats fresh recomputation in the
  regimes we map, and the entire behavior is governed by one deployment knob:
  **coverage** (how much of the original context you keep).
- We locate the memory: it lives in the **downstream-attention trace** — kept tokens
  that attended to the dropped content during prefill carry its imprint. We prove
  this with a controlled position experiment on three multi-hop QA datasets, and
  with a cross-modal experiment where a video-only cache answers audio questions.
- The mechanism converts directly into two zero-training policies: **keep later
  context at a fixed budget** (+3–4 pp), and **escalate coverage only when
  cache-side degeneration signals fire** (same accuracy, 9–22 % less KV — with no
  re-prefill, an option only a reused cache has).

All numbers below are traceable to [`claims_ledger.md`](claims_ledger.md) and
plot-ready in [`figure_data.md`](figure_data.md).

---

## 1. The claim

That a cache can be as good as — sometimes better than — recomputation is not, by
itself, news. The systems community has built an ecosystem (vLLM prefix caching,
LMCache, TurboRAG) around reusing caches for latency, and even "a smaller cache can
*beat* the full one" is by now documented in the eviction literature. But look at
how the field converts that observation into quality, and the recipe is always to
**add machinery**: learned context compression (gist tokens, ICAE,
AutoCompressors), learned retention gates trained precisely because selective
eviction can improve generation ([Make Each Token Count,
2026](https://arxiv.org/abs/2605.09649)), DuoAttention's trained retrieval-head
identification — or, where reuse threatens quality, paying partial recomputation to
patch the cache back toward full-prefill behavior
([CacheBlend](https://arxiv.org/abs/2405.16444)).

Our claim is that the leverage was never in the training. Three things are missing
from the current picture, and they are what this post supplies:

1. **The governing variable.** "Cache ≥ fresh" is *conditional*, and the condition
   is coverage. The full coverage curve has a characteristic shape — a low-coverage
   cliff where the cache is *worse*, a broad mid-range where it wins, and exact
   equality at full coverage — and that shape is the same signature in text and
   video. If you know the curve, you know when reuse helps, when it hurts, and why
   naive sweeps disagree with each other.
2. **The mechanism.** The cache's advantage is a *memory*: kept tokens absorbed
   information from dropped tokens through causal attention during the full
   prefill. This is not a metaphor — we manipulate it causally. Move the evidence
   paragraph to where every kept paragraph can attend to it and the advantage
   maximizes (+2 to +4 pp on all three datasets); move it to where nothing can
   attend to it and the advantage collapses to zero.
3. **The policies.** Because the mechanism is positional and the failure mode is
   detectable at runtime, two training-free methods fall out for free — and one of
   them exploits a structural property only reuse has: escalating coverage costs no
   re-prefill, because the bigger keep-set's rows are already sitting in the stored
   cache.

Everything here runs on stock checkpoints (Qwen3-30B-A3B-Instruct for text,
Qwen3-Omni-30B-A3B Thinker for video) with zero gradient updates. The only thing we
manipulate is which KV rows survive.

## 2. Setup: one prebake, two ways to read it

The regime is **unified-context prebake + subset inference**: a long context (a
multi-paragraph document, a video) is prefilled **once** in full; at use time, only
a subset of its KV entries is retained — because of an eviction budget, a streaming
window, or a relevance filter. The question is what that retained cache is worth,
compared to the natural alternative: re-prefilling the kept subset from scratch.

Every experiment is a paired comparison on identical inputs:

- **fresh** — re-prefill the kept text only. This arm has *never seen* the dropped
  content.
- **reuse (origpos)** — slice the full-context cache down to the kept tokens'
  rows, original positions preserved.
- **reuse (compact)** — same rows, positions re-rotated to be contiguous (the
  convention used by InfLLM/ReKV-style systems). Included so that no result depends
  on a position convention.

And every run carries a built-in **identity gate**: at 100 % coverage nothing is
dropped, the two arms are the same computation, and the measured gap must vanish.
It does — exact to the last item in every accuracy experiment (e.g. .560/.560/.560
on MuSiQue), and zero within noise on the NLL curves (+0.02 ± 0.01) — which is what
separates a mechanism from an instrumentation artifact. Whatever differences appear below
appear *only* when something is deleted, and vanish *exactly* when nothing is.

## 3. The phenomenon: the coverage curve

Sweep coverage and plot the gap (reuse − fresh) on gold-answer NLL. Text
(LongBench multi-hop, n=231):

| coverage kept | reuse − fresh (NLL) | reading |
|---:|---:|---|
| 0 % (question-adjacent chunk only) | **+0.52** | cache *worse* — the cliff |
| 25 % | −0.04 | crossover |
| 50 % | **−0.34** | cache wins — the memory bonus |
| 75 % | −0.09 | shrinking toward… |
| 100 % | +0.02 ≈ 0 | …the identity gate |

Two forces, one knob:

- **(+) the memory bonus.** The reused rows were computed while the full context
  was present; the fresh arm recomputes over the starved subset alone. The bonus is
  largest exactly where recomputation is most starved.
- **(−) the degeneration cost.** A cache over a *tiny* keep-set is a degenerate
  prefix, and decoding over it collapses (the c0 cell hides million-scale
  perplexities). This is keep-set starvation, not a position artifact — the
  compact arm cliffs identically (+0.57 vs +0.52, n.s. difference).

Below the crossover the cost wins; above it the bonus wins; at 100 % both vanish.
**Video shows the same bonus with no cliff** (Qwen3-Omni, EgoSchema n=500: −0.039
at 20 % coverage, p<1e-4, monotone to zero; reproduced in fp32). On Video-MME the
quality difference is large enough to show up bluntly in accuracy: at 10 % coverage,
reuse answers **+8 points** more questions than fresh recomputation overall, **+18
points** on short clips (.33 → .51). Video frames are redundant enough that even
10 % coverage never reaches text-c0 starvation — which is itself evidence that the
cliff is starvation and nothing else.

The practical reading is already useful: *don't over-evict* (the cliff is real),
and in the broad mid-range the reused cache is not a degraded approximation of
recomputation — it is better than it, for free.

## 4. The cache answers questions about deleted content

The coverage curve says the cache knows something fresh recomputation doesn't. The
cleanest way to show *what* is to delete the evidence outright and see who can
still answer.

**Cross-modal version.** Prebake a video **with its audio track**; at use time,
physically delete every audio token's KV (not masked — removed) and keep only the
video rows. Ask audio-dependent questions. On Video-MME (n=597), the audio-prebaked
video cache beats an identically-positioned audio-free prebake by **−0.070 NLL at
full visual coverage (p<1e-4)** — while the audio-free control sits at exactly
0.000, and the gap is invariant to which eviction pattern produced it. The only
difference between the arms is whether audio was *present at prebake*. The video
tokens' KV absorbed it.

**Text version (`drop_gold`).** Multi-hop QA over distractor paragraphs (MuSiQue /
HotpotQA / 2WikiMQA), seeded uniform KV-compression — and now **always physically
remove the answer-evidence paragraph**. The fresh arm has never seen the evidence;
the reuse arm has only the *imprint* of it in surviving rows. Accuracy, n=800 per
dataset: 2Wiki +3.4 pp at cov70 (38 items flip to correct vs 11 the other way,
McNemar p=1e-4) and +2.1 pp even with every other paragraph kept (40:23, p=.04);
MuSiQue's gold-dropped recovery stratum +3.2 pp (27:9, p=.004) and its hardest
4-hop stratum .101 → .129; HotpotQA is directionally positive (+1.6 pp, 28:15,
p=.07). (Absolute numbers are intentionally low — the evidence is gone; the claim
is the paired gap.)

What does recovery look like? Verbatim, drop-gold at 50 % coverage, same kept text
for both arms:

> **Q:** *(3-hop, evidence paragraph removed; gold "11 February 1929")*
> **fresh:** "The question appears to be based on a misunderstanding… there is no
> direct connection…"
> **reuse:** "The author of *Princeps Pastorum* is **Pope John XXIII**, who died in
> Vatican City. Vatican City became an independent country on **11 February
> 1929**, when the Lateran Treaty was signed…"

The cache reconstructs the full three-hop chain, ending in the exact removed date.
In another case the two arms produce *character-identical* answers up to the
missing hop — fresh stops at "Warner Records owns the record label," reuse
continues "…a subsidiary of **Warner Music Group**," which is precisely the removed
parent-company fact.

## 5. The mechanism: a downstream-attention trace

Where, physically, does the recovered information live?

**Hypothesis.** During the full prefill, attention is causal: paragraphs positioned
*after* the evidence attended to it, so their K/V states are functions of it.
Delete the evidence rows and that derived information survives in the rows you
kept. Fresh recomputation never had it. If this is right, the recovery should be
controlled by a purely *positional* quantity: how many kept tokens sit downstream
of the evidence at prebake time.

**Observational check.** Split natural drop-gold items by whether any kept
paragraph sits after the gold paragraph. 2Wiki, full coverage of the remainder:
**+2.9 pp** when downstream attenders exist, **−2.8 pp** when none do.

**Causal test.** Observational splits can be confounded (gold position correlates
with document structure), so we intervene: same question, same kept set, and we
move only the gold paragraph's *prebake position* before deleting it. Gold-FIRST =
every kept paragraph attends to it (maximal trace). Gold-LAST = none can (zero
trace, causal mask). n=714–800 per cell:

| dataset | gold-FIRST (max trace) | gold-LAST (no trace) |
|---|---:|---:|
| HotpotQA | **+4.1 pp** (46:17 flips) | −0.2 pp (7:9) |
| 2Wiki | **+3.6 pp** (55:26) | +0.5 pp (14:10) |
| MuSiQue | **+2.1 pp** (55:38) | +0.9 pp (19:12) |

Maximize the trace and the recovery is largest; eliminate it and the recovery
collapses — on all three datasets, including HotpotQA, which is approximately
*neutral* in natural data and jumps to +4.1 pp the moment the trace is maximized.
The mechanism is universal; natural evidence position merely modulates how much of
it you get. (Per-cell exact tests: HotpotQA-first p=4e-4, 2Wiki-first p=2e-3,
MuSiQue-first p=.10 at n=800 — individually marginal, but the third independent
replication of the same ordering; every gold-LAST cell is null, exactly as the
causal account requires.)

Two further characterizations sharpen the picture:

- **The readout is early.** In the cross-modal setting, swapping only the first 8
  of 48 layers to the cached state recovers 68 % of the full gap, with a plateau by
  layer 24. The trace is consumed in early-to-mid layers — consistent with
  contextual binding, not with some deep-layer answer cache.
- **The trace is a soft prior, not verbatim storage.** Reading 91 recovery
  transcripts: the dominant mode is **disambiguation** — fresh wavers between two
  same-named entities ("…wait, no, that's a mix-up…") and picks the wrong one;
  reuse tips to the right one and proceeds. The model voices the recovered fact as
  its own world knowledge — it cannot cite a paragraph that isn't there. In the
  gain population the recovered facts are *correct*; this is not a hallucination
  channel. The effect size (+2–4 pp, never a verbatim dump) matches: a lossy,
  associative imprint.

## 6. Where it breaks: the honest boundary

A mechanism note is only useful with its sign conditions. The full
three-dataset accuracy matrix (uniform compression, n≈800 each) says:

- **2Wiki:** reuse ≥ fresh at every coverage (+1.1–2.3 pp overall) — the clean win.
- **MuSiQue:** reuse ≥ fresh (+1.3–2.3 pp at cov30–50; the gold-dropped stratum is
  +3.2 pp, 27:9).
- **HotpotQA:** ≈ neutral overall, with **one fresh-favored cell**: gold *kept*,
  50 % coverage, **−1.8 pp** (19:12 — directional, p=.28; the aggregate is small
  and the mechanism is established at case level). Case-level reading: **distractor
  over-anchoring**. The trace amplifies whatever was salient at prebake — including
  a kept, topically-adjacent distractor. In the dump: the gold actor's paragraph is
  removed, a same-titled 1991 film's paragraph is kept; fresh bridges parametrically
  to the right person, reuse confidently answers with the kept film's cast. The
  cell is gone by 70 % coverage, and even on HotpotQA the drop-gold recovery stays
  positive.

The unifying statement: **the trace amplifies what was salient in the full
context.** It helps when the dropped content is the signal you need and the task
resists shortcuts (MuSiQue, 2Wiki); it costs a couple of points when kept
distractors are strong, coverage is low, and the task is shortcut-prone (HotpotQA
2-hop). The sign is (need for the dropped evidence) × (distractor adjacency) — it
is *not* a hop-count story.

And the standing scope condition: all of this requires the **unified-context
prebake**. In vanilla corpus-RAG, where chunks are baked independently, there is no
full-context prefill, hence no trace, hence no asymmetry — we make **no accuracy
claim for generic retrieval-RAG**. The regime that does qualify is common, though:
multi-query sessions over one long document, streaming eviction, agentic loops that
repeatedly subset one long history.

## 7. From explanation to method: two training-free policies

A mechanism you can state positionally is a mechanism you can act on. Both methods
below are zero-training, and the first is uniquely cheap *because* of reuse.

### 7.1 Keep later context at a fixed budget

Direct corollary of causality: later tokens carry the trace of earlier dropped
content; earlier tokens carry nothing of later content. So at a fixed KV budget,
prefer the **last** k paragraphs over the first k. Controlled test (same budget,
same questions, n=800):

| | keep-EARLY gap | keep-LATE gap | diff-in-diff |
|---|---:|---:|---:|
| 2Wiki cov30 | −0.9 pp | **+2.3 pp** (41:23, p=.03) | +3.2 pp |
| 2Wiki cov50 | +0.7 pp | **+4.6 pp** (59:22, p<1e-4) | +3.9 pp |
| MuSiQue cov30/50 | +0.9 / +0.2 pp | +0.6 / +1.0 pp | ≈ 0 |

Fresh accuracy itself is unchanged early-vs-late (.573 vs .576) — so this is not
"late content is more useful"; the benefit is **cache-specific**, exactly as the
mechanism predicts. The discordant-pair signature seals it: under keep-early,
reuse and fresh disagree on almost nothing (19/18 items of 800); under keep-late
they diverge (64/81). The trace in late-kept tokens *is* the difference between
the arms. Implementation cost: one line in your eviction policy. Its scope is
bounded honestly by the MuSiQue row: on 20-paragraph documents with deeper hop
structure the diff-in-diff is ≈ 0 — keep-late pays where the kept tail can attend
to most of what was dropped, and characterizing that condition across corpora is
part of the follow-up.

### 7.2 Degeneration-gated adaptive coverage

The cliff (§3) is the one regime where reuse loses — and it turns out to be
**detectable from the cache side alone**, with no fresh control: abstention
phrasing, n-gram repetition, and no-EOS truncation in the low-coverage generation
predict wrong answers everywhere (on MuSiQue at cov30, P(wrong | signal) = .80 vs
.57 without). Policy: answer at low coverage; escalate coverage only while a signal
fires.

Simulated on the real sweep generations (a POC, stated as such): the gate matches
the accuracy of the best fixed coverage with **9–22 % less KV** on all three
datasets, and on 2Wiki it strictly dominates fixed cov50 (+1.5 pp accuracy *and*
−9 % budget). The oracle ceiling is striking — .922 at 42 % of the full budget,
versus .865 for always-full — so the signal design has headroom.

**Why this is a reuse method and not a generic trick:** keep-sets are nested, and
the stored full-document cache already contains every row. Escalation is therefore
*fetch more rows and re-decode* — **no re-prefill**. Under fresh computation,
every escalation is a full re-prefill of a longer context. Adaptive coverage is
uniquely cheap in exactly the regime this post is about.

### 7.3 Outlook: trace-aware eviction

Importance-based eviction (H2O, SnapKV) keeps what is attended-to. The mechanism
suggests an orthogonal axis: a chunk whose information has already been **absorbed**
by surviving downstream tokens is safe to evict *regardless of its importance* —
its trace remains. An absorption/redundancy criterion is computable from prebake
attention. We state it here as the open direction and benchmark it in a follow-up.

## 8. What we ruled out

Negative results that constrain the interpretation, briefly:

| Competing story | Verdict |
|---|---|
| "Position-preserving reuse beats compaction" | **Null.** Re-rotating kept rows costs +0.006 NLL, n.s., even under maximal M-RoPE shear. The win is the memory, not the position convention. |
| "This beats ReKV/InfLLM-style systems" | **No.** A faithful ReKV reimplementation ties plain reuse. We claim the mechanism and the policies, not a systems win. |
| "Cache ≥ fresh is universal" | **No.** The text low-coverage cliff is real, convention-independent, and is starvation. |
| "The benefit/penalty tracks hop-count" | **No.** Sign tracks (need for dropped evidence) × (distractor adjacency); HotpotQA flips to +4.1 pp when the trace is maximized. |
| "The cross-modal trace lives in deep layers" | **No.** Early–mid readout: first 8 layers carry 68 % of the gap. |

## 9. Positioning

- **TurboRAG / prefix-caching / LMCache** treat reuse as a TTFT problem and aim for
  quality-*neutrality*. The quality dimension mapped here is invisible in that
  framing.
- **The cache-fusion line** ([CacheBlend](https://arxiv.org/abs/2405.16444),
  CacheClip, A³) runs in the *opposite* direction: chunks are baked **piecewise**,
  so their KV lacks cross-attention to surrounding text, and the methods spend
  selective recomputation to *restore* it. Our setting is the mirror image — the
  cross-attention is already there, baked in full context, and we show it is not a
  cost to patch but an asset to exploit: it is precisely what makes the reused
  cache *better* than recomputation. Same physics, opposite sign.
- **Learned compression** (gist tokens, ICAE, AutoCompressors) and **learned
  retention** (DuoAttention; [Make Each Token Count,
  2026](https://arxiv.org/abs/2605.09649)) buy smaller caches with training — the
  latter comparing against the *full-cache* baseline. The coverage curve says a
  large fraction of the quality is available with no training at all; and the
  paired reuse-vs-fresh design (same kept tokens, only the KV source differs)
  isolates *why* — a trained compressor inherits, rather than explains, the trace.
- **InfLLM / ReKV** (block retrieval + compaction): compatible; our results say
  their position-compaction convention is harmless, and their retrieved blocks
  carry the same memory.
- **MuKV-style token selection**: its advantage decomposes to fine, largely
  query-driven token selection — not prebakeable; orthogonal and composable with
  everything here.
- **H2O / SnapKV** and their information-theoretic descendants (e.g. CapKV): all
  score *which tokens to keep* by importance or predictive capacity. Absorption —
  whether a token's information already survives in downstream kept rows — is the
  axis none of them measure; that is §7.3.
- **Multimodal KV work** (AccKV, MEDA, AudioKV; HERMES's cache-as-memory framing
  for streaming) optimizes eviction across modalities for efficiency. None tests
  modality-absent recovery — whether a deleted modality's information survives in
  the other modality's KV. As of this writing (June 2026) we find no occupant for
  the combination presented here: the conditional quality map, the causal
  mechanism, the cross-modal recovery demonstration, and mechanism-derived
  training-free policies.

## 10. Takeaways

**If you run inference:** position-preserving reuse is the correct default; in
multi-query / streaming regimes over one long context the accuracy bonus is free.
Don't over-evict — the cliff is real. At a fixed budget, keep *later* context.
Gate coverage on cache-side degeneration signals; under reuse, escalation costs no
re-prefill.

**If you do research:** KV reuse is a quality lever, not just a latency one. The
downstream-attention trace is causally established, early-read, lossy, and behaves
like a soft disambiguating prior — it amplifies whatever was salient at prebake,
which sets both its value and its failure mode. Absorption-aware eviction is open.

---

*Models: Qwen3-30B-A3B-Instruct-2507 (text), Qwen3-Omni-30B-A3B Thinker (video,
M-RoPE), stock checkpoints, no fine-tuning. Datasets: LongBench
2WikiMQA/HotpotQA/MuSiQue; original MuSiQue ans-v1.0 dev, HotpotQA dev-distractor,
2WikiMQA dev; EgoSchema-Subset; Video-MME. Text accuracy cells n=796–800 with
greedy decoding, alias-match scoring, persisted generations; NLL sweeps n=231–1500;
video n=236–597 with paired Wilcoxon; every run carries the cov100 identity gate;
bf16 with fp32 control. Accuracy gaps are not a verbosity artifact of alias-match
scoring: reuse generations are the *same length or slightly shorter* than fresh in
every dataset × coverage cell (−3 to +1 words on average), and scoring is paired. Tables: [`figure_data.md`](figure_data.md); claim-by-claim
evidence: [`claims_ledger.md`](claims_ledger.md); instrument:
`scripts/49_musique_hop.py`.*
