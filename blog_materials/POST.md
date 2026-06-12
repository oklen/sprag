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
  this with controlled position experiments on three multi-hop QA datasets and
  **two model families** (Qwen, Mistral — the Mistral effects are larger), and with
  a cross-modal experiment where a video-only cache answers audio questions. A
  position ablation rules out primacy/lost-in-the-middle as the explanation, and a
  **counterfactual probe** shows the trace carries verbatim content the model
  cannot know any other way — at low bandwidth.
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
   maximizes (+2 to +6 pp, on three datasets and two model families); move it to
   where nothing can attend to it and the advantage collapses to zero. A
   counterfactual probe then shows the trace can carry *verbatim content* — the
   cache reproduces a fabricated fact that exists nowhere except in the deleted
   tokens' KV imprint.
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

To preempt the most common mis-mapping: even when the context is assembled from
many source documents (our QA items concatenate ~10–20 Wikipedia paragraphs,
retrieval-style), the assembly is prefilled **jointly, as one sequence** — the
paragraphs cross-attend at bake time. We never stitch together KV that was baked
in *different* contexts; that piecewise-baked regime (TurboRAG / CacheBlend-style
chunk caches) lacks the cross-attention this post is about, and none of our
accuracy claims apply to it (§6, §9). The query is appended after the cache and is
identical in both arms.

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

Sweep coverage and plot the gap (reuse − fresh) in **answer accuracy** — greedy
generation, alias-match, paired items, n≈800 per cell:

| coverage kept | MuSiQue (Qwen) | 2Wiki (Qwen) | 2Wiki (Mistral) | HotpotQA (Qwen) |
|---:|---:|---:|---:|---:|
| 10 % | −0.5 pp | +1.1 pp | +0.9 pp | +0.6 pp |
| 30 % | **+2.3 pp** (43:25, p=.04) | **+2.3 pp** (38:19, p=.02) | +1.8 pp (p=.098) | −0.3 pp |
| 50 % | +1.3 pp | +1.1 pp | **+2.8 pp** (p=.009) | −0.7 pp |
| 70 % | +0.1 pp | +1.2 pp | **+2.4 pp** (p=.008) | +0.5 pp |
| 100 % | 0 — exact | 0 — exact | 0 — exact (0:0) | 0 — exact |

The same shape, three datasets, two model families: a bonus that peaks at mid
coverage, fades toward full coverage, and lands on **exact zero** at 100 % — the
identity gate. (HotpotQA hovers near zero overall; §6 maps why.) On video the
effect is blunter still: at 10 % visual coverage, reuse answers **+8 points** more
Video-MME questions than fresh recomputation, **+18 points** on short clips
(.33 → .51).

Two forces, one knob:

- **(+) the memory bonus.** The reused rows were computed while the full context
  was present; the fresh arm recomputes over the starved subset alone. Split the
  items by whether the answer-evidence paragraph survived the cut and the bonus
  concentrates exactly where theory puts it — in the **evidence-dropped stratum**
  (MuSiQue cov30: +3.2 pp, 27:9, p=.004), where the cache is the only place any
  imprint of the evidence still lives.
- **(−) the degeneration cost.** Push coverage low enough and the cache becomes a
  degenerate prefix. In accuracy this surfaces in the **evidence-kept stratum at
  10 % coverage** — the one place fresh recomputation still has everything it
  needs while the cache's context has collapsed: MuSiQue .608 → .519 (**−8.9 pp**,
  cache worse), 2Wiki −2.3 pp, Mistral −3.5 pp. The runtime symptoms are visible
  in the generations themselves — abstention phrasing, n-gram repetition,
  truncation — which is what makes the cliff *detectable and actionable* (§7.2).
  (At the pathological extreme of ~0 % coverage the collapse is total — our
  earliest NLL instrument recorded million-scale perplexities there, identical for
  position-preserving and re-rotated caches: starvation, not a position artifact.)

Below the crossover the cost wins; above it the bonus wins; at 100 % both vanish.
Video never reaches that starvation regime even at 10 % coverage (frames are
redundant), and accordingly shows the bonus with **no cliff at all** — which is
itself evidence that the cliff is starvation and nothing else.

The practical reading is already useful: *don't over-evict* (the cliff is real,
and it lives precisely where the evidence survived but little else did), and in
the broad mid-range the reused cache is not a degraded approximation of
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

| | gold-FIRST (max trace) | gold-LAST (no trace) |
|---|---:|---:|
| **Qwen3-30B-A3B** | | |
| HotpotQA | **+4.1 pp** (46:17 flips, p=4e-4) | −0.2 pp (7:9) |
| 2Wiki | **+3.6 pp** (55:26, p=.002) | +0.5 pp (14:10) |
| MuSiQue (n=2400) | **+2.4 pp** (148:91, p=3e-4) | −0.0 pp (45:46) |
| **Mistral-Small-24B** | | |
| HotpotQA | **+5.8 pp** (56:10, p<1e-4) | −0.5 pp (1:5) |
| 2Wiki | **+3.9 pp** (60:29, p=.0013) | −0.2 pp (2:4) |

Maximize the trace and the recovery is largest; eliminate it and the recovery
collapses — five out of five dataset × family cells, every FIRST cell individually
significant (exact McNemar), every LAST cell null. This includes HotpotQA, which is
approximately *neutral* in natural data and jumps to +4–6 pp the moment the trace
is maximized: the mechanism is universal; natural evidence position merely
modulates how much of it you get. And it is not a quirk of one architecture — the
dense, non-Qwen Mistral shows *larger* effects than the MoE it replicates. A
final signature: under gold-LAST the two arms barely disagree *at all* (1:5, 2:4
discordant items of 800) — remove the trace and reuse and recompute become the
same model.

**Is it the position, not the attention?** Gold-FIRST could be suspected of a
primacy or attention-sink advantage — early tokens might simply be encoded more
strongly, and "lost in the middle" is the ready-made counter-story. The ablation:
place gold at a seeded *random interior slot* and record how many kept paragraphs
sit downstream. Three findings (both families, n=800/cell). (1) Aggregate middle
recovery sits between first and last, as a dose account predicts. (2) **Slot 0 is
not special**: gold at slots 1–2 — with 7–8 of 9 paragraphs downstream — recovers
**+4.9 pp in both families**, as much as gold-first itself. The predictor is
downstream mass, not the privileged first position. (3) The dose-response is
threshold-shaped rather than linear, and a stratification explains why: recovery
concentrates on items where the **semantically bound supporting paragraph** sits
downstream of gold (Mistral 2Wiki: +2.5 pp, 18:7, p=.043, vs −0.6 pp when every
supporting paragraph precedes it; Qwen directionally the same). The trace is not a
diffuse field — it pays when it lands on the tokens that are *about* the dropped
content. (This is also exactly why keeping *later* context works as a policy —
§7.1.)

**Does the trace store content, or just prime retrieval?** The sharpest objection
to "the cache remembers what you deleted": maybe the trace merely nudges the model
toward facts it already knows parametrically — Vatican City's 1929 is in the
pretraining data, after all. The probe: rewrite the gold paragraph's answer year to
a **fabricated** one before the prebake (eligibility-filtered so the year appears
nowhere else), then drop the paragraph and ask. Three controls anchor the readout
(2Wiki, n=461): with the counterfactual paragraph *visible*, the model echoes the
fabricated year 93% of the time (it trusts context over parameters; arms exactly
identical — the identity gate holds for this pipeline too). With the paragraph
dropped: the cached arm reproduces the **fabricated** year where fresh never does —
**6:0 one-sided flips at max trace (exact p=.031)**, dose-consistent (2:0 at
natural position) — while the **true**-year channel shows no significant difference
(27:23): the priming story loses, the content story wins. The texture is telling:
the cache often gets the fabricated *year* right but the day or month wrong
("April 15, 1946" where the document said "November 10, 1946") — a
partial-fidelity imprint, with net verbatim bandwidth around **1–2 % per readable
item** (a year-token lower bound; MuSiQue's null is exactly what this bandwidth
times its much lower task ceiling predicts — the probe's sensitivity is
ceiling × bandwidth). One transcript even cites the ghost: *"This is supported by
the text provided"* — said of a paragraph that no longer exists.

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
| HotpotQA cov30 | −0.1 pp | **+2.0 pp** (46:30, p=.085) | +2.1 pp |
| HotpotQA cov50 | −0.5 pp | +0.4 pp | +0.9 pp |
| MuSiQue cov30/50 | +0.9 / +0.2 pp | +0.6 / +1.0 pp | ≈ 0 |

Fresh accuracy itself is unchanged early-vs-late (.573 vs .576) — so this is not
"late content is more useful"; the benefit is **cache-specific**, exactly as the
mechanism predicts. The discordant-pair signature seals it: under keep-early,
reuse and fresh disagree on almost nothing (19/18 items of 800); under keep-late
they diverge (64/81). The trace in late-kept tokens *is* the difference between
the arms. Implementation cost: one line in your eviction policy. Scoreboard,
honestly: strong on 2Wiki, weak-positive on HotpotQA at the tighter budget,
≈ 0 on 20-paragraph MuSiQue — **keep-late never hurts** (it is the keep-early arm
that goes negative) and pays where the trace is strong, i.e. where the kept tail
can attend to most of what was dropped — the same bridge-routing condition §5
established.

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

### 7.3 Outlook: the rolling agent context

The deployment regime where we think this matters most is not one bake + many
queries — it is the **rolling agent context**: user–agent and agent–agent turns
accumulate, the window fills, and something has to go. Today's frameworks truncate
the oldest turns and re-prefill the survivors, or selectively evict tokens by
importance. The results above reorder those options:

- **Drop-oldest is the principled policy, and slicing beats re-prefilling.**
  Keep-late *is* "truncate the oldest turns" — except you keep the surviving KV
  rows instead of recomputing them from the surviving text. Those rows were baked
  while the dropped turns were still present, so they carry the absorbed trace;
  re-prefilling from text is the one way to actually lose it. The usual
  correctness instinct — "the context changed, re-prefill to be safe" — picks the
  arm that is both slower *and* worse.
- **Selective mid-context eviction needs a query that doesn't exist yet.**
  Importance scoring (H2O, SnapKV) is query-driven; in an agent loop the future
  queries are unknown, and query-free importance selection adds roughly nothing
  (we measured this directly in the video decomposition). With no query, recency
  is the only signal available — and it happens to be the trace-optimal one.
- **Summarize-then-evict is an engineered trace.** Generate the summary while the
  old turns are still in context, append it, *then* drop them: the summary
  tokens' KV absorbs the full context at bake time, exactly like any other
  downstream token. The natural trace and the summary are then the same kind of
  object at different bandwidths — the free channel carries gist and binding at
  ~1–2 % verbatim bandwidth (§5), the summary carries whatever you spend decode
  tokens on. The bandwidth measurement doubles as a spec for what the summary
  must contain: verbatim detail and long-chain structure (what the free channel
  drops), not gist (what it already keeps).

One honest gap: everything in this post is measured on a single bake and a single
slice. A rolling context is bake → evict → extend → evict again, and whether
absorption compounds or decays across cycles is unmeasured — that is the
experiment we run next, alongside **trace-aware eviction**: a chunk whose
information has already been absorbed by surviving downstream tokens is safe to
evict *regardless of its importance*. An absorption/redundancy criterion is
computable from prebake attention; we state it here as the open direction and
benchmark it in a follow-up.

## 8. What we ruled out

Negative results that constrain the interpretation, briefly:

| Competing story | Verdict |
|---|---|
| "Position-preserving reuse beats compaction" | **Null.** Re-rotating kept rows costs +0.006 NLL, n.s., even under maximal M-RoPE shear. The win is the memory, not the position convention. |
| "This beats ReKV/InfLLM-style systems" | **No.** A faithful ReKV reimplementation ties plain reuse. We claim the mechanism and the policies, not a systems win. |
| "Cache ≥ fresh is universal" | **No.** The text low-coverage cliff is real, convention-independent, and is starvation. |
| "The benefit/penalty tracks hop-count" | **No.** Sign tracks (need for dropped evidence) × (distractor adjacency); HotpotQA flips to +4.1 pp when the trace is maximized. |
| "The cross-modal trace lives in deep layers" | **No.** Early–mid readout: first 8 layers carry 68 % of the gap. |
| "Gold-first wins by primacy / attention-sink / lost-in-the-middle" | **No.** Random interior slots 1–2 recover as much as slot 0 (+4.9 pp, both families); the predictor is downstream mass, concentrated where the supporting paragraph is downstream. |
| "The trace just primes parametric retrieval" | **No.** Counterfactual probe: fabricated-year reproduction 6:0 (p=.031) while the true-year channel is null. |
| "It's a Qwen quirk" | **No.** Curve and causal A/B replicate on Mistral-Small-24B with larger effects. |

## 9. Positioning

- **TurboRAG / prefix-caching / LMCache** treat reuse as a TTFT problem and aim for
  quality-*neutrality*. The quality dimension mapped here is invisible in that
  framing.
- **The cache-fusion line** ([CacheBlend](https://arxiv.org/abs/2405.16444),
  CacheClip, A³; most recently [RelayCaching](https://arxiv.org/abs/2603.13289),
  which transplants one agent's *decode-phase* KV into the next agent's prefill)
  runs in the *opposite* direction: the KV was computed under a **different or
  missing context** than the one it is used in, so the absorbed-prefix component is
  contamination, and the methods spend selective recomputation to remove it. Our
  setting is the mirror image — the cache is baked in the **same** full context the
  query is about, and the absorbed component is not a cost to patch but the asset
  itself: it is precisely what makes the reused cache *better* than recomputation.
  One sentence resolves the apparent contradiction: **whether KV "deviation" from
  recompute is an error or a memory depends on whether the bake-time context is the
  context the query wants to condition on.** Notably, RelayCaching's own
  measurements independently confirm the physics we exploit: the deviation (= the
  absorbed context) is sparse, concentrated on a few token positions, layer-wise
  non-uniform — and their repair criterion even scores tokens by *downstream
  attention influence*, the same quantity our causal experiments manipulate.
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

**If you build agents:** when the history must shrink, slice the cache instead of
re-prefilling the kept turns — the surviving KV remembers part of what you
dropped, and recomputing it from text is the only way to lose that. Drop
oldest-first (recency is the trace-optimal query-free signal), and write the
summary *before* you evict, so its KV bakes in the turns it summarizes.

**If you do research:** KV reuse is a quality lever, not just a latency one. The
downstream-attention trace is causally established on two model families, routed
through semantically bound downstream tokens rather than positions, early-read,
and lossy — it demonstrably carries verbatim content (counterfactual probe) at
roughly 1–2 % bandwidth, while behaving day-to-day like a soft disambiguating
prior. It amplifies whatever was salient at prebake, which sets both its value and
its failure mode. Absorption-aware eviction is open.

---

*Models: Qwen3-30B-A3B-Instruct-2507 (text), Mistral-Small-24B-Instruct-2501
(cross-family replication; family-adapted prompt conventions, same instrument and
seeds), Qwen3-Omni-30B-A3B Thinker (video, M-RoPE) — stock checkpoints, no
fine-tuning. Datasets: LongBench 2WikiMQA/HotpotQA/MuSiQue; original MuSiQue
ans-v1.0 dev, HotpotQA dev-distractor, 2WikiMQA dev; EgoSchema-Subset; Video-MME.
Text accuracy cells n=796–800 (MuSiQue gold-position cells n=2400) with greedy
decoding, alias-match scoring, persisted generations; gold position manipulated
{first, random-interior, last} with downstream counts recorded; counterfactual
probe on eligibility-filtered year-answer items (n=461/226), scored by word-bounded
year-token match for both the fabricated and the true year; NLL sweeps n=231–1500;
video n=236–597 with paired Wilcoxon; every run carries the cov100 identity gate;
bf16 with fp32 control. Accuracy gaps are not a verbosity artifact of alias-match
scoring: reuse generations are the *same length or slightly shorter* than fresh in
every dataset × coverage cell (−3 to +1 words on average), and scoring is paired.
Tables: [`figure_data.md`](figure_data.md); claim-by-claim evidence:
[`claims_ledger.md`](claims_ledger.md); instruments: `scripts/49_musique_hop.py`
(Qwen), `scripts/50_xfam_hop.py` (cross-family), `scripts/51_counterfactual.py`
(counterfactual probe).*
