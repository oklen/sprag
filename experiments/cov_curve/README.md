# The coverage curve is the mechanism's signature — and the gain is *conditional*

This documents what the **cached-vs-fresh coverage curve** actually looks like
across its full range on text, and a 3-arm experiment isolating the role of the
RoPE **position convention** (position-preserving vs compaction). It extends
[`../coverage_sinkdup/README.md`](../coverage_sinkdup/README.md) (mechanism +
sink-dup fix) and the video curves on the `video-kv-omni` branch.

**TL;DR**
1. The gap `cached − fresh` is not a single number; it is a **curve that bends with
   how much context you keep**. That shape *is* the evidence for the information-gap
   mechanism — and it falsifies artifact explanations (it converges to **exact zero**
   at full coverage).
2. **"cached ≥ fresh" is NOT universal.** On text there is a **low-coverage cliff**
   where cached is *worse* than fresh (degeneration). The bonus only appears once
   enough context is retained. Any conclusion holds only within a regime.
3. The cliff is driven by **keep-set starvation, not the position convention**:
   position-preserving (origpos) reuse cliffs just as hard as compaction. The
   convention is a *second-order* effect that modestly favors origpos in the
   mid-coverage bonus regime.

## Two competing forces

The cached-reuse gain is the net of two opposing effects, and coverage is the knob
that trades them off:

- **(+) memory bonus** — cached KV were built over the *full* document, so they
  carry a trace of the dropped chunks; fresh recomputes over the starved subset
  only. This *favors cached*, and grows the **more** you drop (low coverage).
- **(−) fragment / degeneration cost** — a cache assembled over a *tiny* keep-set is
  a degenerate prefix; decode collapses (repetition / open-`<think>` / garbage,
  PPL → 1e6+). This *hurts cached*, worst at the extreme low end.

At extreme low coverage (−) dominates → **cliff**. In the mid range (+) takes over →
**cached beats fresh**. At full coverage both → 0 → **curves meet exactly**.

## Curve (A3B-Instruct-2507, LongBench 2wikimqa/hotpotqa/musique, n=231)

3-arm scorer `scripts/33_origpos_3arm.py` (= the sink-dup-fixed `scripts/36_a3b_fix.py`
plus `build_origpos_cache` / `gold_nll_origpos` / `gen_origpos`), all three arms
scored on the **same** documents in one pass. Metric = gold-answer NLL = mean
log-PPL; **gap vs fresh, lower is better** (negative = cached wins). `agg_3arm.py`.

| coverage | n | NLL fresh | **origpos − fresh** (SEM) | **compact − fresh** (SEM) | acc f / o / c |
|---:|---:|---:|:--|:--|:--|
| **c0** (sink+target only) | 231 | 13.675 | **+0.519** (.122) | **+0.571** (.118) | .68 / .68 / .69 |
| c25 | 231 | 14.011 | **−0.042** (.105) | +0.101 (.096) | .67 / .71 / .70 |
| c50 | 211 | 13.175 | **−0.341** (.091) | −0.211 (.088) | .66 / .68 / .67 |
| c75 | 206 | 12.627 | −0.093 (.056) | −0.021 (.054) | .66 / .67 / .69 |
| **c100** | 181 | 12.388 | **+0.019** (.011) | **+0.019** (.011) | .67 / .66 / .66 |

Reading the curve:
- **c0 cliff:** both arms are **worse** than fresh by ~+0.5 nats (million-scale PPL =
  degeneration). cached loses here.
- **c25 → c50:** crossover; the memory bonus takes over (origpos already beats fresh
  at c25; both clearly win by c50).
- **c100 identity gate:** origpos and compact are **bit-identical** to each other
  (`+0.019` each), and ≈ fresh. With nothing dropped, the two conventions coincide
  and there is nothing to remember → the gain vanishes **exactly**. This is the
  built-in control: an artifact would not converge to zero here.

(The compact arm reproduces the earlier 2-arm run: c0 `compact−fresh = +0.571` here
vs `+0.581` in `data/a3b_cov_fix.s*` — faithful port.)

### Contrast with video (no cliff)

On Qwen3-Omni / Video-MME (`video-kv-omni` branch, n=236) the curve is **monotone**:
cached beats fresh already at the lowest coverage (cov10 `−0.229`) and converges to
0 at cov100, with **no** low-coverage cliff. Reason: video cov10 still retains the
attention sink plus several temporal groups — it never reaches the text-c0 level of
starvation. The cliff is a property of *how starved* the keep-set is, not of modality.

## origpos vs compaction — the cliff is convention-independent

Two ways to place the surviving tokens' RoPE positions at reuse time:
- **origpos** (position-preserving, our default): kept tokens keep their **original,
  gappy** absolute positions; cached K reused verbatim; pairwise relative geometry =
  prebake.
- **compaction** (InfLLM / ReKV style): kept tokens repacked to **contiguous**
  positions; K un-rotate→re-rotate (`shift_rope`) → relative geometry compressed.

We expected the cliff to be a **compaction position-scramble** effect (crushing the
relative geometry of a few far-apart survivors). It is **not**:

- **At c0 the cliff is essentially identical** for both: origpos `+0.519` vs compact
  `+0.571`. The difference (0.052 nats) is **within one SEM (n.s.)**. Position-
  preserving reuse degenerates just as hard. ⇒ **the cliff = keep-set starvation, not
  the convention.** (Consistent with the video deep-dive #3 origpos-vs-compact NULL.)
- **In the mid-coverage bonus regime origpos is consistently the better convention:**
  it already beats fresh at c25 (`−0.042`) while compaction still loses (`+0.101`),
  and gives a larger bonus at c50 (`−0.341` vs `−0.211`) and c75 (`−0.093` vs
  `−0.021`). Direction-consistent ~+0.10–0.14 nats favoring origpos across all three
  mid covs (~1–1.5σ each → **modest, not dramatic**).

**Takeaway:** keep position-preserving reuse as the default (it never hurts and
modestly helps in the bonus regime), but do **not** claim it rescues the low-coverage
cliff — nothing in the position convention does. The honest scope of "cached ≥ fresh"
is: a mid-to-high coverage statement; below the starvation threshold the sign flips.

## Files

- `scripts/33_origpos_3arm.py` — 3-arm (origpos / compact / fresh) coverage scorer +
  greedy-gen accuracy; `--mode coverage`. cov100 is a 3-arm identity gate.
- `experiments/cov_curve/agg_3arm.py` — pooled NLL (mean log-PPL) + acc + cliff check
  per coverage over `data/a3b_cov_3arm.s*.json`.
- `experiments/cov_curve/agg_text_cov.py` — same for the earlier 2-arm
  `data/a3b_cov_fix.s*.json`.
- Raw shard JSON is gitignored; all aggregated numbers are in this README.
