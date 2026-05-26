"""MK-derived MAGS calibration.

Builds (T+, T-) pairs from the MK suite where:
  T+ = assemble the chunk that *actually* contains the gold needle, then
       prompt tail "\\n\\nQ: ... \\nA:" — captures the residual when the
       model has the right needle in context.
  T- = assemble chunks holding *same-template distractor* needles (no
       gold), with the same prompt tail — captures the residual when
       retrieval gave the model "almost right" but actually wrong chunks.

This replaces the prior 04_calibrate_mags.py strategy of "bottom-K
cosine wrong chunks" — see NOTES.md §8 "MAGS calibration data quality"
for why that didn't work.

Inputs: an MK suite (from scripts/data/gen_mk_suite.py).
Output: a .pkl loadable by sprag.mags.calibrate.load_mags.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

import torch
from safetensors.torch import load_file

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import build_chunk_cache, load_meta
from sprag.embed import JinaEmbedder
from sprag.assemble import ChunkPlacement, patched_full_attn, make_inv_freq_for
from sprag.mags.calibrate import (
    grab_last_residual, fit_mags, save_mags, DEFAULT_MAGS_LAYERS,
)
from gen_niah import NEEDLES  # type: ignore


def reconstruct_needle_text(template_id: int, picks: dict) -> str:
    tmpl = NEEDLES[template_id][0]
    return tmpl.format(**picks)


def find_chunk_for_needle(cache_dir: Path, meta: dict, tok, needle_text: str) -> int:
    """Return chunk_id whose decoded text contains the most distinctive
    substring of the needle. We use a window around the middle of the
    needle string because chunks split mid-sentence."""
    # take a ~50-char "spine" from the middle of the needle, robust to
    # different chunk boundaries
    n = len(needle_text)
    spine = needle_text[max(0, n // 2 - 25): n // 2 + 25].lower()
    for c in meta["chunks"]:
        full = tok.decode(load_file(str(cache_dir / f"chunk_{c['id']:05d}.safetensors"))["input_ids"])
        if spine in full.lower():
            return c["id"]
    # fall back: any chunk whose preview contains the first 30 chars
    head = needle_text[:30].lower()
    for c in meta["chunks"]:
        if head in c["text_preview"].lower():
            return c["id"]
    return -1


def build_placement(cache_dir, chunk_ids, chunk_lookup, prefix_len: int = 0):
    placements = []
    flat_ids: list[int] = []
    cursor = prefix_len
    for cid in chunk_ids:
        t = load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))
        ids = t["input_ids"]
        L = int(ids.shape[0])
        cached = {li: (t[f"K_l{li}"], t[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(chunk_lookup[cid]["a_start"]),
            b_start=cursor, length=L, cached=cached,
        ))
        flat_ids.extend(ids.tolist())
        cursor += L
    return placements, flat_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True,
                    help="dir from scripts/data/gen_mk_suite.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--k_svd", type=int, default=4)
    ap.add_argument("--tau_quantile", type=float, default=0.95)
    ap.add_argument("--max_neg_chunks", type=int, default=3,
                     help="Cap on number of distractor chunks per T-")
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]
    print(f"Calibrating MAGS on MK suite: {len(case_ids)} cases")

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    emb = JinaEmbedder()
    inv_freq = make_inv_freq_for(model).to(device)

    pos_bucket = {li: [] for li in DEFAULT_MAGS_LAYERS}
    neg_bucket = {li: [] for li in DEFAULT_MAGS_LAYERS}

    skipped = {"single_template": 0, "missing_chunk": 0}
    for ci in case_ids:
        cd_src = args.suite / f"case_{ci:02d}"
        haystack = (cd_src / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd_src / "queries.jsonl").open()]

        cache_dir = args.out.parent / f"_mk_calib_case{ci:02d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        build_chunk_cache(model, tok, haystack, cache_dir,
                          chunk_size=args.chunk_size, embed_fn=emb.encode_passage)
        meta = load_meta(cache_dir)
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        # Resolve each query's needle to its chunk_id.
        for q in queries:
            needle_text = reconstruct_needle_text(q["template_id"], q["picks"])
            gold_chunk = find_chunk_for_needle(cache_dir, meta, tok, needle_text)
            if gold_chunk < 0:
                skipped["missing_chunk"] += 1
                continue

            # Find sibling-needle chunks (same template, different picks).
            distractor_chunks: list[int] = []
            for q_other in queries:
                if q_other["id"] == q["id"]:
                    continue
                if q_other["template_id"] != q["template_id"]:
                    continue
                other_needle = reconstruct_needle_text(q_other["template_id"], q_other["picks"])
                cid = find_chunk_for_needle(cache_dir, meta, tok, other_needle)
                if cid >= 0 and cid != gold_chunk and cid not in distractor_chunks:
                    distractor_chunks.append(cid)
            if not distractor_chunks:
                skipped["single_template"] += 1
                continue
            distractor_chunks = distractor_chunks[: args.max_neg_chunks]

            prompt_tail_ids = tok("\n\nQ: " + q["question"] + "\nA:",
                                   add_special_tokens=False).input_ids

            # T+: only the gold chunk
            pos_placements, pos_flat = build_placement(cache_dir, [gold_chunk], chunk_lookup)
            pos_input = torch.tensor([pos_flat + prompt_tail_ids], dtype=torch.long, device=device)
            with torch.no_grad(), patched_full_attn(model, pos_placements, inv_freq=inv_freq), \
                    grab_last_residual(model) as cap:
                model.model(pos_input, use_cache=False)
            for li in DEFAULT_MAGS_LAYERS:
                pos_bucket[li].append(cap[li])

            # T-: distractor chunks only (same template, wrong picks)
            neg_placements, neg_flat = build_placement(cache_dir, distractor_chunks, chunk_lookup)
            neg_input = torch.tensor([neg_flat + prompt_tail_ids], dtype=torch.long, device=device)
            with torch.no_grad(), patched_full_attn(model, neg_placements, inv_freq=inv_freq), \
                    grab_last_residual(model) as cap:
                model.model(neg_input, use_cache=False)
            for li in DEFAULT_MAGS_LAYERS:
                neg_bucket[li].append(cap[li])

            print(f"  case {ci} q{q['id']} t{q['template_id']}: "
                  f"gold={gold_chunk} distractors={distractor_chunks}")

    n_pos = len(pos_bucket[DEFAULT_MAGS_LAYERS[0]])
    n_neg = len(neg_bucket[DEFAULT_MAGS_LAYERS[0]])
    print(f"\nCollected {n_pos} T+ and {n_neg} T- pairs   "
          f"(skipped: {skipped})")

    if n_pos == 0 or n_neg == 0:
        print("Not enough data for SVD; aborting")
        return

    pos = {li: torch.stack(v, 0) for li, v in pos_bucket.items()}
    neg = {li: torch.stack(v, 0) for li, v in neg_bucket.items()}

    params = fit_mags(pos, neg, k=args.k_svd, tau_quantile=args.tau_quantile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_mags(params, args.out)
    print(f"Saved MAGS params to {args.out}")
    for li in params.layer_indices:
        print(f"  layer {li}: B shape={tuple(params.B[li].shape)}  "
              f"tau={params.tau[li]:.3f}")


if __name__ == "__main__":
    main()
