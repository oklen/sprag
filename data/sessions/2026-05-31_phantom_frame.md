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

| mode | frame K/V | chunk K/V | acc | McNemar vs raw | avg_tok |
|------|-----------|-----------|-----|----------------|---------|
| `rfresh` | fresh | **fresh** | 81.0 | p=0.28 tied | 2503 |
| `rframe` | fresh | cached α1 | 80.7 | p=0.38 tied | 2503 |
| `cframe` | **cached** α1 | cached α1 | 80.3 | p=0.47 tied | 2503 |
| `raw` | — (no frame) | fresh | 78.3 | — | 1291 |
| `splice_a1` | — (no frame) | cached α1 | 73.3 | p=0.049 ↓ | 1291 |

Pairwise McNemar among the real-frame variants: `rframe` vs `cframe` p=1.0,
`rframe` vs `rfresh` p=1.0, `cframe` vs `rfresh` p=0.85 — **all three tied**, and
all tied with `raw`.

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
- **Efficiency caveat:** as MEASURED, `cframe` still forwards the frame+chunk
  tokens (overwritten), so avg_tok=2503 — same cost as `rframe`. The prefill-skip
  is only realised with a true inject-KV path (forward sink+query only). And even
  then **indep (§5ab) is cheaper** (~1300 tok, no frame, also tied with raw).
  `cframe`'s value is mechanistic (drift = isolation), not a new efficiency win.
