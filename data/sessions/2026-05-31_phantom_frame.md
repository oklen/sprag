# Phantom-context frame experiments — 2026-05-31

Standalone results + reproduction for the phantom-context-frame line of work.
Companion in NOTES: **§5ad** (MK probe + follow-ups) and **§5ad-RGB** (RGB).
Memory: `sprag-splice-decomp` (TL;DR), this file linked from `MEMORY.md`.

## Question

Keep the **full cache** (chunk K/V built in `[anchor][real preceding doc][chunk]`
— the "phantom context" is baked into the chunk's hidden states). At USE time,
does the cached chunk just need *a* frame in front of it to route attention, or
does it need *real semantic* context? And does whatever works transfer from the
synthetic single-needle MK suite to real multi-chunk RAG (RGB)?

Scripts:
- `scripts/20_phantom_frame.py` — MK probe (single gold chunk, k=1).
- `scripts/16_rgb_eval.py` — RGB, modes `gframe_topk`/`gfront_topk`/`rframe_topk`
  /`cframe_topk`/`rfresh_topk` (+ `build_frame_placements`,
  `build_cframe_placements`, `build_rfresh_flat`, `reconstruct_doc_tokens`,
  `GENERIC_FRAME_TEXT`).

```bash
# MK probe (5-arm + 9-arm follow-ups are the same script, default arms)
.venv/bin/python scripts/20_phantom_frame.py --out data/phantom_frame_mk.json    # real/rother/rand/ph/shift
.venv/bin/python scripts/20_phantom_frame.py --out data/phantom_frame_mk2.json   # + fsent/sflu16/64/256/sph64
# RGB
.venv/bin/python scripts/16_rgb_eval.py --out data/rgb_frame.json  --resume \
    --modes raw_topk splice_topk_a1 gframe_topk gfront_topk rframe_topk
.venv/bin/python scripts/16_rgb_eval.py --out data/rgb_frame2.json --resume \
    --modes raw_topk splice_topk_a1 rframe_topk cframe_topk rfresh_topk
```

Construction (each chunk placed back near its build geometry): `b_start =
M + len(frame)`, `a_start = canon` (build position); cached K is RoPE-shifted by
`b_start - canon`. Empty frame = standard splice; length-`a0` frame = position
restored (delta 0); length-`W` frame = position-shifted to `M+W`.

---

## MK probe (k=1 single gold, n=60, `data/phantom_frame_mk*.json`)

Arms differ ONLY in what fills the length-`a_start` frame the query reads.
`real` reconstructs the exact build prefix ⇒ fresh K/V == cache ⇒ splice is a
no-op ⇒ doubles as a faithfulness sanity + full-reprefill upper bound.

| arm | frame content | restored? | /60 |
|-----|---------------|-----------|-----|
| `rother` | a DIFFERENT doc's leading tokens (fluent, irrelevant) | yes | **57** |
| `sflu256` | 256 coherent tokens, chunk **shifted** | shifted | **57** |
| `real` | the chunk's TRUE preceding context | yes | 54 |
| `sflu64` | 64 coherent, shifted | shifted | 47 |
| `rand` | random token ids (gibberish), same length | yes | 46 |
| `sflu16` | 16 coherent, shifted | shifted | 43 |
| `ph` | repeated `<|endoftext|>` placeholder | yes | 43 |
| `fsent` | a fluent SENTENCE tiled to length `a0` | yes | 42 |
| `sph64` | 64 placeholder, shifted | shifted | 41 |
| `shift` | no frame (standard splice at M) | — | 24 |

McNemar (exact, paired):

| pair | totals | discordance | p | verdict |
|------|--------|-------------|---|---------|
| `real` vs `rother` | 54 vs 57 | 0 / 3 | 0.25 | tied (real never uniquely wins) |
| `rand` vs `ph` | 46 vs 43 | 6 / 3 | 0.51 | tied |
| `fsent` vs `ph` | 42 vs 43 | 6 / 7 | 1.0 | tied (repeated sentence == placeholder) |
| `sflu256` vs `rother` | 57 vs 57 | 3 / 3 | 1.0 | tied (short shifted == full restored) |
| `sflu64` vs `sflu256` | 47 vs 57 | 1 / 11 | 0.006 | length matters |
| `real` vs `ph` | 54 vs 43 | 12 / 1 | 0.003 | coherent > degenerate |
| `rother` vs `ph` | 57 vs 43 | 15 / 1 | 0.0005 | coherent > degenerate |
| `ph` vs `shift` | 43 vs 24 | 20 / 1 | <1e-4 | any frame ≫ no frame |

**MK conclusions.** (1) Frame *presence* is the big lever (shift 24 → frame
43–57). (2) The content tier is **coherent, non-repetitive natural language**
(real/rother/sflu256 ≈ 55) ≫ degenerate/incoherent (rand/ph/fsent ≈ 44);
**relevance is irrelevant** (rother ≈ real) and a repeated sentence collapses to
placeholder level (fsent ≈ ph). (3) **Position restoration is unnecessary** — a
~256-tok coherent frame + a normal RoPE-shift ties the full-length restored
ceiling (sflu256 = rother = 57).

---

## RGB (300 records, Jina top-5, `standard`=full cache, run 1 `data/rgb_frame.json`)

All frame modes splice at **α=1.0 (pure cached** — the prefill-skip regime):
prepend a 256-tok FRESH frame, splice the chunk after it. `gframe` = generic
coherent passage before each chunk; `gfront` = generic, first chunk only;
`rframe` = each chunk's OWN real preceding 256 doc tokens.

| mode (α=1.0) | acc | McNemar vs raw | avg_tok |
|------|-----|----------------|---------|
| `raw_topk` (fresh, no splice) | 78.3 | — | 1291 |
| **`rframe`** (real preceding-256) | **80.7** | p=0.38 **tied** | 2503 |
| `splice_a1` (no frame) | 73.3 | p=0.049 (sig ↓) | 1291 |
| `gfront` (generic, front only) | 69.3 | p=0.0005 (sig ↓) | 1547 |
| `gframe` (generic, per-chunk) | 65.3 | p<1e-4 (sig ↓) | 2571 |

Parroting (output contains the generic frame text): `gframe` **42/300 (14%)**,
`gfront` 6/300 — the mechanism of the generic-frame collapse.

**RGB conclusions — REVERSES MK.** (1) The generic-frame trick is **net-negative**
on real multi-chunk RAG (gframe 65.3 < no-frame splice 73.3): open-ended
generation **parrots** the irrelevant frame, and multi-chunk chunks already have
coherent neighbours, so a generic frame is redundant disruption. "Relevance is
irrelevant" was an MK artifact of the constrained single-needle answer. (2) The
chunk's **OWN real preceding context** works and rescues the α=1.0 splice footgun
(73.3 → 80.7, tied with fresh raw). **Caveat:** rframe is ~2× tokens (real frame
fed fresh) ⇒ ties raw accuracy at *higher* cost; indep (§5ab: 76.7/74.7, tied
with raw at *lower* cost) stays the better path.

---

## RGB run 2 — cache-reuse + 2×-fresh control (`data/rgb_frame2.json`)

Two questions: **#1** the real preceding 256 tokens ARE the previous chunk (cache
tiles the doc on 256 boundaries) → splice the previous chunk's K/V as the frame
too (`cframe`, fully cached, prefill-skip candidate); **#2** the "2× fresh"
control (`rfresh`): same `[real-frame][chunk]` layout but EVERYTHING fresh — does
rframe's 80.7 come from the cache/splice or simply from more real context?

indep rows are from the separate `data/rgb/rgb_indep.json` run
(`cache_kind=indep`, §5ab-RGB): SAME 300 records/seed (its `raw_topk` matches
this run's 300/300), so paired against this run's `raw`.

| mode | cache | frame K/V | chunk K/V | acc | McNemar vs raw | avg_tok |
|------|-------|-----------|-----------|-----|----------------|---------|
| `rfresh` | std | fresh | **fresh** | 81.0 | p=0.28 tied | 2503 |
| `rframe` | std | fresh | cached α1 | 80.7 | p=0.38 tied | 2503 |
| `cframe` | std | **cached** α1 | cached α1 | 80.3 | p=0.47 tied | 2503 |
| `raw` | — | — | fresh | 78.3 | — | 1291 |
| `indep` α0.5 | indep | — | 0.5·cached | 76.7 | p=0.47 tied | ~1300 |
| `v_only` (indep) | indep | — | fresh K, cached V | 76.0 | — | ~1300 |
| `indep` α1.0 | indep | — | cached α1 | 74.7 | p=0.14 tied | ~1300 |
| `splice_a1` | std | — | cached α1 | 73.3 | p=0.049 ↓ | 1291 |
| `k_only` (indep) | indep | — | cached K, fresh V | 69.3 | — | ~1300 |

Pairwise McNemar among the real-frame variants: `rframe`≡`cframe` p=1.0,
`rframe`≡`rfresh` p=1.0, `cframe`≡`rfresh` p=0.85 — all three tied, all tied
with `raw`. **`cframe` (80.3) vs indep α1.0 (74.7): p=0.0213 — cframe
SIGNIFICANTLY beats indep in the pure-α=1.0 regime** (cframe+33 / indep+16).

**Conclusions.**
- **#1 cache-reuse is LOSSLESS.** `cframe` (80.3) ≡ `rframe` (80.7), p=1.0:
  splicing the previous chunk's *cached* K/V as the frame == computing it fresh.
  So a **fully-cached α=1.0 splice** (frame + chunk both cached) ties fresh `raw`
  (p=0.47) — the prefill-skip path is viable on the 6 full-attn layers.
- **#2 the lift is the real CONTEXT, not the cache mechanism.** `rfresh`
  (everything fresh, 81.0) ≡ `rframe` ≡ `cframe`: the splice contributes nothing
  ±. And `rfresh` ≈ `raw` (p=0.28) — even fresh real-context doesn't beat plain
  raw. **The "+2.4" over raw was noise; raw/rframe/cframe/rfresh are all ≈79%.**
- **Net:** giving each cached chunk its real local context (= its free cached
  predecessor) CURES the α=1.0 splice footgun losslessly (73.3 → 80.3, fresh
  parity) but gives **no accuracy gain over raw**. It pins the standard-cache
  drift footgun (§5u/§5w) on chunk ISOLATION — restore the cached neighbour and
  α=1.0 splicing is fine.
- **cframe vs indep — a cost/accuracy trade, not a clean winner.** In the pure
  α=1.0 (prefill-skip) regime, `cframe` 80.3 **significantly beats** indep α1.0
  74.7 (p=0.02) — restoring the cached predecessor as context lifts the
  standard-cache splice above the isolation-built indep cache. indep is CHEAPER
  (no frame; ~1300 tok even as-measured vs cframe's 2503) and ties raw at α=0.5
  (76.7), but α<1 admixes fresh K (not a clean prefill-skip), and at pure α=1.0
  indep sits at 74.7. So: indep = cheaper, slightly lower; cframe = higher α=1
  accuracy at ~2× cached injection (and as-MEASURED still forwards the overwritten
  frame+chunk tokens — the real speedup needs a true inject-KV path that forwards
  only sink+query). Neither beats fresh `raw` on accuracy; the cached splice is
  free-at-best, as everywhere else in the project.

---

## Chunk-size sweep — the footgun is a SHORT-chunk artifact (`data/rgb_frame_cs{512,1024}.json`)

User hypothesis (2026-05-31): if a FIXED short frame (256) still cures the α=1.0
splice for LONG chunks, the frame overhead (`frame_len/chunk_size`) shrinks and
the method becomes meaningful. Test: fix `frame_len=256`, sweep `chunk_size`.
`cframe` generalised to splice the previous chunk's LAST `frame_len` cached
tokens (a fixed frame for any chunk length). All α=1.0, n=300, Jina top-5.

| chunk_size | frame overhead | raw | splice_a1 (no frame) | cframe | rframe | footgun raw−splice |
|------------|----------------|-----|----------------------|--------|--------|--------------------|
| 256  | 100% | 78.3 | 73.3 | 80.3 | 80.7 | **+5.0** (p=0.049) |
| 512  | 50%  | 84.3 | 78.0 | 80.3 | 81.0 | **+6.3** (p=0.0019) |
| 1024 | 25%  | 82.3 | **85.0** | 85.7 | 86.0 | **−2.7** (p=0.15, gone) |

McNemar `cframe` vs `splice_a1`: cs256 p=0.0019 (frame CURES), cs512 p=0.32
(frame only partial — cframe 80.3 < raw 84.3), cs1024 p=0.81 (no diff, nothing
to cure).

**Conclusion — the hypothesis flips to something better.** The α=1.0 splice
footgun is a SHORT-chunk artifact: footgun +5.0 (cs256) → +6.3 (cs512) → −2.7
(cs1024, GONE). The drift (§5w) lives in the chunk's first few boundary tokens
(built with preceding context absent at assembly); for a long chunk those are a
tiny fraction → the chunk self-mitigates. So at `chunk_size≥1024` the PURE cached
splice with NO frame (`splice_a1` 85.0) already ties/nominally-beats fresh `raw`
(82.3, p=0.15); `cframe` adds nothing (p=0.81). **At long chunks the frame is
UNNECESSARY, not just cheap** — and it only cleanly "cures" at cs256 where its
overhead is 100% (a useless operating point); at cs512 it only partially helps.
**Practical prefill-skip recipe: chunk_size ≥ ~1024, pure α=1.0, NO frame.**
Bonus: at cs1024 cached ≥ fresh (85.0 vs 82.3) — first hint the ReAttention
premise pays off (a long chunk's cache carries the FULL real doc context that the
short-assembly fresh forward never sees). Caveat: `raw` peaks at cs512 (84.3) =
retrieval sweet-spot; splice-free@1024 comes with raw below its 512 peak
(coverage vs splice-cleanliness trade).

Chunking is fully STATIC fixed-length (`split_into_chunks` = `range(0, n,
chunk_size)`, non-overlapping, no semantic/passage-boundary awareness; RGB doc =
`"\n\n".join(shuffled passages)`, so fixed chunks straddle passage boundaries).
Open direction: semantic/passage-aligned chunking to attack retrieval recall —
the real ceiling (§5u: raw 78.3 ≪ full-context 86.3).

---

## Head-drift test — answer-position bucketing (cs=1024, `data/rgb_frame_cs1024_ap.json`)

Hypothesis (user): the cs=1024 footgun vanished only because the answer rarely
sits in the chunk's drift-prone HEAD; a frame would be needed for robustness if
it did. Test: log each answer's earliest position within its retrieved chunks
(`find_answer_position` → `ans_frac`), bucket splice_a1 (no frame) vs cframe
(frame) vs raw. n=300, 0 retrieval miss.

Surprise: answers concentrate at the head — 208/300 (69%) at frac<0.10, 262/300
(87%) at frac<0.25 (RGB passages are answer-early; ans_frac = min over retrieved
chunks). So the head bucket is huge → a powerful test.

| ans_frac bucket | n | raw | splice_a1 | cframe | rframe |
|-----------------|---|-----|-----------|--------|--------|
| head 0–10% | 208 | 182 | **187** | **187** | 189 |
| 10–25% | 54 | 43 | 43 | 43 | 43 |
| 25–50% | 24 | 16 | 17 | 18 | 17 |
| 50–75% | 11 | 6 | 8 | 8 | 8 |
| tail 75–100% | 3 | 0 | 0 | 1 | 1 |

HEAD (frac<0.25, n=262): splice vs cframe 7/7 p=1.0 (tied); raw vs splice 7/12
p=0.36 (splice nominally higher). TAIL (frac≥0.5, n=14): splice vs cframe 0/1.

**Head-drift REFUTED.** Even with the answer AT the chunk head (262/300), the
no-frame cached splice ≡ the framed version (187=187) and ≥ fresh raw — at EVERY
position bucket. The frame contributes nothing anywhere, including the head. So:
my "long chunk self-mitigates" (too aggregate), the user's "answer dodges the
head" (answers are mostly IN the head, yet fine), and "frame as head-insurance"
(head needs no frame) are all wrong. **Drift magnitude is set by build-vs-use
CONTEXT distance (§5w), not token-position-in-chunk.** At cs=1024 top-5 covers
~5/9 of the doc → each cached chunk's build context (full doc) ≈ its assembly
use context → even head tokens' preceding context is ~restored → drift is small
everywhere. (At cs=256 top-5 = 5/36 → use ≪ build → whole chunk drifts.) Also
confirms the MK position-uniformity (drift independent of needle position) and
why cached ≥ fresh (cache carries full-doc context the short assembly lacks).
