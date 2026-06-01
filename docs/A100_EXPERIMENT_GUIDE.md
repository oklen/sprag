# sprag — Large-Model Experiment Guide (8×A100)

**Goal of this run:** take the cache-splicing / coverage research to a *capable*
model. The dev model (Qwen3.5-0.8B) is a **short-context single-fact extractor** —
it cannot reason and drowns in long context — which caps the accuracy story. A
7B–70B model removes that ceiling, so we can finally land the **ACC + PPL
coverage curve** on real data and test whether **cache reuse can match or beat
fresh recompute** (the "global-context bonus").

This document is self-contained. Read §1–§3 for *why*, §4 for *what to run*, §5
for *how to implement on a fresh box*, §6 for commands & deliverables.

---

## 1. The research object (one paragraph)

RAG systems reuse a chunk's KV-cache instead of re-encoding it (TurboRAG /
ReAttention style). A chunk's cache is built while the model sees the *whole
document*; at use time it is spliced into a *short* assembly of retrieved chunks.
Two opposing effects:
- **Drift cost** — the cached K/V "remembers" a build-time context that the short
  assembly contradicts → splice can hurt (§5w drift law: cost = build-vs-use
  context distance).
- **Global-context bonus** — that same memory carries document context the short
  fresh assembly never sees → splice can *help*.

**`coverage`** = fraction of a chunk's true preceding context present in the
assembly. We want the **Δ(coverage)** curve and the crossover **c\*** where reuse
becomes lossless / beneficial.

## 2. What we already established (dev model, Qwen3.5-0.8B)

- **§5w drift law:** splice cost is governed by build-vs-use context distance, not
  a K–V binding rule. K drifts more than V.
- **cos(K/V) drift curve, n=388** (`data/coverage_drift.json`, `scripts/22_coverage_drift.py`):
  cosK rises 0.888→0.961→0.977→0.989→0.9998 over coverage 0→100%. 73% of the gap
  to 1.0 closes in the first 25% of coverage; the low-coverage drift is *entirely*
  the target's first token (cosK_first 0.46→0.96 by cov25); cosV>cosK; drift is
  mid-network. **This is high-power but treats fresh as a ceiling — cos can never
  show cache>fresh.** That is why we need an accuracy-space metric.
- **Lossless reuse on RGB:** isolation-built (`indep`) caches and
  previous-chunk-as-frame (`cframe`) tie fresh; at chunk_size 1024 cached even
  nominally beat fresh (85.0 vs 82.3) — the first global-context-bonus hint.
- **Capability ceiling (the reason for this run)** — gate-probes, full-context vs
  no-context, dev model:

  | task | type | full-ctx acc | gap |
  |------|------|-------------|-----|
  | RGB (top-5 retrieval, ~1.5k) | single-hop, short | ~78–86% | large |
  | LongBench multifieldqa_en (~8k) | single-hop, long | 17.5% | +17.5 |
  | QuALITY (MC) | comprehension | ~chance 26% | +1 |
  | 2wikimqa / hotpotqa | multi-hop | ~20% | +2.5 |

  → the 0.8B can extract a *stated* fact but cannot reason or use long context.
  The bonus needs the model to *combine* global context with local info — exactly
  what it can't do. **A capable model is the unlock.**

## 3. The metric: gold-answer PPL (use this, not 0/1 accuracy alone)

Compute the model's **perplexity of the gold answer** given the assembly:

```
NLL(gold | assembly) = -(1/|a|) Σ_t log P(a_t | assembly, a_<t)
PPL = exp(NLL)
```

Why PPL is the right backbone:
1. **Continuous → high power.** Every question yields a real number; no
   answerability gate, no starved n (the n=58 problem on RGB disappears).
2. **Behavioral, not representational.** Unlike cos, it is about the *answer*.
3. **Not bounded by fresh.** `PPL_cached` can be *lower* than `PPL_fresh` → the
   bonus shows up as a graded effect even when thresholded accuracy can't see it.

**Validated on the dev model** (gold-NLL full→no-context):
- multifieldqa_en: PPL **2.5 → 20.1**, context lowers gold PPL in **100%** of Qs.
- 2wikimqa: PPL **27.5 → 44.3**, lowers in **58%** of Qs (modest on the weak model;
  expected to be strong on a capable one).

Report **both** PPL and 0/1 accuracy on the big model (accuracy becomes usable
once the model can reason). PPL is the powered backbone; accuracy is the headline.

---

## 4. Experiment plan (prioritized)

All experiments share: a single coherent document, fixed-length chunking
(chunk_size 256 or 512), a target chunk, and a coverage knob.

**Coverage protocol (per question):**
- `target` = a deep chunk that causally summarizes the doc. Two valid choices:
  (a) the chunk containing the answer string (locate by alias match) — preferred
  when locatable; (b) the last chunk — when not. Use (a) for these datasets (the
  answer string is present in the gold passage).
- For coverage `c ∈ {0,25,50,75,100}%`: include the **last c% of the target's
  preceding chunks** (contiguous), then the target. Prepend a short sink
  (first M=4 doc tokens). Question + gold answer appended last.
- **Two arms, identical token layout:**
  - **cached** — the included chunks + target take their K/V from the **full-doc**
    forward (each "remembers" the omitted earlier doc). α = 1.0 (pure cached).
  - **fresh** — the same tokens recomputed seeing only what's in the assembly.
- Metrics per cell: `NLL(gold)`, `PPL`, `0/1 acc` (generation + alias match).

### E1 — Big-model context-value & capability re-gate *(no cache surgery; do first)*
Confirm the chosen model actually reasons and uses context. Run the §3 gate
(full-context vs no-context: PPL **and** accuracy) on multifieldqa_en, 2wikimqa,
hotpotqa, musique, QuALITY, RGB. **Gate to proceed:** full-context accuracy high
(>60% single-hop, >40% multi-hop) with a wide clean gap. This validates datasets
and gives the fresh-arm coverage baseline. Pure HuggingFace, ~1 GPU.

### E2 — Cached-vs-fresh coverage curve *(THE core experiment)*
The §4 protocol, both arms, on the gated datasets. Deliver per dataset + pooled:
- `PPL_fresh(c)` and `acc_fresh(c)` — context value vs coverage.
- `ΔPPL(c) = PPL_cached − PPL_fresh` and `Δacc(c)` — splice cost/benefit in answer
  space. **Lossless ⇒ Δ≈0; drift ⇒ Δ>0 (PPL) / Δ<0 (acc); bonus ⇒ the reverse.**
- **c\*** = smallest coverage where cached ties fresh (paired test n.s.).
- **Bonus check:** at low coverage, is `PPL_cached < PPL_fresh` / `acc_cached >
  acc_fresh`? Strongest on **multi-hop** (the bridge fact lives in omitted context
  the cache still carries) — this is where cache>fresh should appear.

### E3 — cos(K/V) drift curve at scale
Port `scripts/22_coverage_drift.py` to the big model: does cosK 0.89→1.0 / "first
token drifts most" / cosV>cosK / mid-network hold at 7B–70B? Pairs with E2
(representation parity by ~cov25 ⇒ does PPL/acc parity arrive at the same c\*?).

### E4 — Bonus stress test *(optional, highest upside)*
On multi-hop, force the bonus regime: ensure the answer chunk is included but the
**bridge/supporting** chunk is *omitted* at low coverage. If `acc_cached >
acc_fresh` there, the cache demonstrably supplies reasoning context the fresh
assembly lacks — the headline "cache beats fresh" result. (HotpotQA/2Wiki gold
`supporting_facts` from the *original* releases give exact bridge locations; the
LongBench versions drop them, so either re-download originals or approximate the
bridge by the non-answer supporting sentence.)

---

## 5. Implementation on a fresh 8×A100 box

### 5.1 Environment
```bash
git clone git@github.com:oklen/sprag.git && cd sprag
python -m venv .venv && . .venv/bin/activate
pip install "torch>=2.4" transformers accelerate datasets
# flash-linear-attn / causal-conv1d only needed for Qwen3.5-hybrid; NOT needed
# if you pick a pure full-attention model (recommended below).
pip install flash-attn --no-build-isolation   # optional, speeds attention
```

### 5.2 Model choice — **use a pure full-attention model**
The dev model is a *hybrid* (6 full-attn + 18 GatedDeltaNet layers); the
linear-attn state is not position-sliceable, so splicing only ever skipped 6/24
layers (§5z). **Pick a standard full-attention instruct model** (Qwen2.5-7B/14B/
32B/72B-Instruct, Llama-3.1-8B/70B-Instruct, etc.). Then:
- **every** layer's KV is cacheable/spliceable → full prefill-skip, cleaner result;
- no GatedDeltaNet, no `fla`/`causal-conv1d`, no linear-state folding;
- standard RoPE (full rotary) → cache reuse needs **no custom RoPE-shift code**
  if you use the position-preserving method below.

With 8×A100 (80GB) you can run up to 72B with `device_map="auto"`; 7B–14B fits on
one GPU and lets you parallelize datasets across the 8 cards.

### 5.3 Cache reuse — the portable, model-agnostic recipe
**Do NOT port sprag's `patched_full_attn` / `shift_rope` (they are hardcoded to
Qwen3.5-0.8B's RoPE: head_dim 256, partial_rotary_factor 0.25, θ=1e7, full-attn
layers (3,7,11,15,19,23)).** Instead splice at the standard HF cache level using
**position-preserving reuse** — no re-rotation needed:

1. **Build:** one forward over the full document with `use_cache=True`; keep the
   returned `past_key_values` (a `DynamicCache`) and the per-token `position_ids`
   (0..N-1). This is the full-doc cache; every chunk's K/V is rotated at its
   *original* absolute position.
2. **Cached arm at coverage c:** construct a cache holding only the
   [sink ∪ selected chunks ∪ target] token slices, **kept at their original
   positions** (gaps preserved). Append the question + gold tokens fresh. Pass
   explicit `position_ids` (original positions for cached tokens, continuing for
   the question). Because RoPE is relative, the target sees the included chunks at
   their true relative distances; the omitted chunks are simply absent. **No
   un-rotate/re-rotate** — the cached K/V is reused verbatim.
3. **Fresh arm:** the same selected token ids re-encoded from scratch (seeing only
   the selected chunks), then question + gold. For a clean paired comparison, give
   the fresh tokens the *same* original `position_ids` as the cached arm.
4. **Score:** run the model over [assembly + gold]; take logits at the gold-token
   positions; `NLL = -mean log_softmax(logits)[gold]`. For accuracy, `generate`
   ~24 tokens and alias-match.

This isolates exactly "cached-K/V (built over full doc) vs fresh-K/V (built over
assembly)" at identical positions — the splice effect. α=1.0 only (HF cache is
per-position all-or-nothing); the α<1 blend was a dev-model micro-finding and is
not needed for the headline.

> Reference implementations (dev model, custom splice): `scripts/21_coverage_curve.py`
> (ACC), `scripts/22_coverage_drift.py` (cos), `src/sprag/assemble.py`
> (`patched_full_attn`, `shift_rope`). Read them for the protocol logic; reimplement
> the splice via the HF-cache recipe above for the big model.

### 5.4 Sanity gate before any real run (critical)
- **α=0 / no-op:** cached arm at coverage 100% with the *full* cache must produce
  **identical** logits to a plain full-document forward (max abs logit diff < 1e-3
  in fp32, looser in bf16). If not, the cache surgery is wrong — fix before
  proceeding. This is the single most important check.
- **PPL scorer sanity:** trivial context "Maria's favorite color is purple." →
  gold "purple" must have far lower NLL than distractors; with no context it
  reverts to prior. (The dev-model scorer passed this.)

### 5.5 PPL scorer (reference, framework-agnostic)
```python
def gold_nll(model, tok, prefix_ids, position_ids, past_kv, answer_str, device):
    oid = tok(" " + answer_str, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix_ids + oid], device=device)
    pos = torch.tensor([position_ids + list(range(position_ids[-1]+1,
                        position_ids[-1]+1+len(oid)))], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, position_ids=pos,
                       past_key_values=past_kv, use_cache=False).logits[0]
    rows = logits[len(prefix_ids)-1: len(prefix_ids)+len(oid)-1].float()
    lp = torch.log_softmax(rows, -1)
    return -sum(lp[j, t].item() for j, t in enumerate(oid)) / len(oid)
```
**Memory note:** never `.float()` the full (T × vocab) logits — slice the gold
positions first (a 0.8B OOM'd a 16GB card by float-ing 7k×151k logits). On A100s
it matters less but slice anyway.

### 5.6 Datasets (already in repo under `data/benchmarks/longbench_v1/data/`)
`multifieldqa_en.jsonl` (single-hop long, the clean +PPL signal), `2wikimqa.jsonl`
(multi-hop, best length: median 6.4k, 169/200 <12k), `hotpotqa.jsonl`,
`musique.jsonl`. Schema: `{input, context, answers[list], length}`. `context` is
the concatenated passages; `answers` is the gold alias list. RGB lives in
`data/benchmarks/rgb/`. For E4's gold bridge locations, re-download original
HotpotQA/2Wiki (the LongBench copies drop `supporting_facts`).

### 5.7 Stats
Paired per question: McNemar (accuracy) and Wilcoxon signed-rank / paired-t on NLL
(PPL). Report mean ± SEM and the paired sign rate ("% Qs where cached < fresh").
n in the hundreds per dataset is easily affordable.

---

## 6. Commands & deliverables

Write the big-model runner as `scripts/30_bigmodel_coverage.py` (new; uses the
HF-cache recipe). Suggested CLI:
```bash
export SPRAG_MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
# E1 gate
python scripts/30_bigmodel_coverage.py --mode gate \
    --data 2wikimqa multifieldqa_en hotpotqa musique --limit 100 \
    --out data/big_gate.json
# E2 core coverage curve (PPL + ACC, cached vs fresh)
python scripts/30_bigmodel_coverage.py --mode coverage \
    --data 2wikimqa multifieldqa_en --coverages 0 25 50 75 100 \
    --chunk_size 256 --limit 200 --out data/big_coverage.json
# E3 cos drift at scale (port of 22)
python scripts/30_bigmodel_coverage.py --mode drift \
    --data 2wikimqa --limit 200 --out data/big_drift.json
```

**Report back (JSON + a short markdown table):**
1. E1 gate table: per dataset, full vs no-context PPL **and** accuracy (+ the gap).
   Confirms the model is capable enough to proceed.
2. E2 curves: per dataset + pooled — `PPL_fresh(c)`, `ΔPPL(c)`, `acc_fresh(c)`,
   `Δacc(c)`, with SEM; the crossover **c\***; and the **bonus verdict** (is
   cached < fresh / cached > fresh anywhere, with paired p-values?).
3. E3: cosK/cosV/cosK_first vs coverage — does the dev-model drift law scale?
4. Any cell where **cache beats fresh** (PPL or acc) — the headline. Note dataset,
   coverage, effect size, p-value.

**Expected story if it works:** on a capable model, multi-hop shows `acc_cached >
acc_fresh` at low coverage (cache supplies the omitted bridge), giving the first
clean "reuse beats recompute" result; ΔPPL/Δacc → 0 by some c\* that matches the
cos-drift convergence (~cov25). That closes the loop from representation (cos) →
answer likelihood (PPL) → accuracy, and quantifies the global-context bonus.

---

*Maintainer context lives in `NOTES.md` (§5w, §5ad, §5ad-DRIFT) and the session
docs under `data/sessions/`. The dev-model memory file
`sprag-model-capability-ceiling` explains why this run uses a large model.*
