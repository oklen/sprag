"""Harvest-T- MAGS calibration.

Reads an MK eval result JSON, classifies each query as T+ (gold needle
retrieved and model answered correctly) or T- (gold retrieved but
model produced a distractor or other / degenerate output), then
re-replays each query's assembled prefill — using exactly the chunks
the runner picked at eval time — and captures the last-token residual.
SVD over those gives a subspace that points at "I have the right
chunks but I'm about to drift", not "I'm in a bookshop query."

Usage:
  scripts/10_calibrate_mags_harvest.py \
    --suite data/mk/suite_8k \
    --results data/mk/mk_8k_results.json \
    --top_k 6 \
    --out data/mags/mags_mk_harvest.pkl
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
    return NEEDLES[template_id][0].format(**picks)


def find_chunk_for_needle(cache_dir: Path, meta, tok, needle_text: str) -> int:
    spine = needle_text[max(0, len(needle_text)//2 - 25): len(needle_text)//2 + 25].lower()
    for c in meta["chunks"]:
        full = tok.decode(load_file(str(cache_dir / f"chunk_{c['id']:05d}.safetensors"))["input_ids"])
        if spine in full.lower():
            return c["id"]
    head = needle_text[:30].lower()
    for c in meta["chunks"]:
        if head in c["text_preview"].lower():
            return c["id"]
    return -1


def build_placement(cache_dir, chunk_ids, chunk_lookup, prefix_len=0):
    placements, flat = [], []
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
        flat.extend(ids.tolist())
        cursor += L
    return placements, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True,
                    help="MK eval result JSON from scripts/07_run_mk_niah.py")
    ap.add_argument("--top_k", type=int, default=6,
                    help="which reattn_kN bucket from the result to harvest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--k_svd", type=int, default=4)
    ap.add_argument("--tau_quantile", type=float, default=0.95)
    args = ap.parse_args()

    results = json.loads(args.results.read_text())
    bucket_key = f"reattn_k{args.top_k}"
    print(f"Harvesting from {bucket_key} in {args.results}")

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    emb = JinaEmbedder()
    inv_freq = make_inv_freq_for(model).to(device)

    pos_bucket = {li: [] for li in DEFAULT_MAGS_LAYERS}
    neg_bucket = {li: [] for li in DEFAULT_MAGS_LAYERS}
    counts = {"pos": 0, "neg": 0,
              "skip_no_gold_chunk": 0, "skip_gold_not_retrieved": 0,
              "skip_no_bucket": 0}

    for case in results["cases"]:
        ci = case["id"]
        cd_src = args.suite / f"case_{ci:02d}"
        haystack = (cd_src / "haystack.txt").read_text()

        cache_dir = args.out.parent / f"_harvest_case{ci:02d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        build_chunk_cache(model, tok, haystack, cache_dir,
                          chunk_size=args.chunk_size, embed_fn=emb.encode_passage)
        meta = load_meta(cache_dir)
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        # The eval result JSON drops 'picks'; re-load from the suite's queries.jsonl
        # for the picks → needle reconstruction.
        queries_src = {x["id"]: x for x in (json.loads(l) for l in (cd_src / "queries.jsonl").open())}

        for q in case["queries"]:
            if bucket_key not in q:
                counts["skip_no_bucket"] += 1
                continue
            qb = q[bucket_key]
            qs = queries_src[q["id"]]
            needle_text = reconstruct_needle_text(qs["template_id"], qs["picks"])
            gold_chunk = find_chunk_for_needle(cache_dir, meta, tok, needle_text)
            if gold_chunk < 0:
                counts["skip_no_gold_chunk"] += 1
                continue
            retrieved = qb.get("retrieved", [])
            if gold_chunk not in retrieved:
                counts["skip_gold_not_retrieved"] += 1
                continue

            cls = qb["class"]
            target_bucket = pos_bucket if cls == "correct" else neg_bucket
            label = "pos" if cls == "correct" else "neg"

            prompt_tail_ids = tok("\n\nQ: " + qs["question"] + "\nA:",
                                   add_special_tokens=False).input_ids
            placements, flat = build_placement(cache_dir, retrieved, chunk_lookup)
            inp = torch.tensor([flat + prompt_tail_ids], dtype=torch.long, device=device)
            with torch.no_grad(), patched_full_attn(model, placements, inv_freq=inv_freq), \
                    grab_last_residual(model) as cap:
                model.model(inp, use_cache=False)
            for li in DEFAULT_MAGS_LAYERS:
                target_bucket[li].append(cap[li])
            counts[label] += 1
            print(f"  case {ci} q{q['id']} t{qs['template_id']} cls={cls:10s} "
                  f"-> {label}  retrieved={retrieved}")

    print(f"\nHarvest done.  {counts}")
    if counts["pos"] == 0 or counts["neg"] == 0:
        print("Not enough data; aborting"); return

    pos = {li: torch.stack(v, 0) for li, v in pos_bucket.items()}
    neg = {li: torch.stack(v, 0) for li, v in neg_bucket.items()}
    params = fit_mags(pos, neg, k=args.k_svd, tau_quantile=args.tau_quantile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_mags(params, args.out)
    print(f"Saved -> {args.out}")
    for li in params.layer_indices:
        print(f"  layer {li}: B shape={tuple(params.B[li].shape)}  tau={params.tau[li]:.3f}")


if __name__ == "__main__":
    main()
