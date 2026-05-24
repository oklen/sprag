"""Offline MAGS calibration.

Construct (T+, T-) trajectory pairs for a set of NIAH cases:
  T+: oracle ReAttention — retrieve exactly the chunk containing the needle.
  T-: anti-retrieval — pick bottom-K chunks (no needle) and force them into
      the assembled context, so the model is forced to drift.

Capture residual-stream activations at the last query token for layers 11/15/19,
SVD-fit the error subspace, save params.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from safetensors.torch import load_file

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import build_chunk_cache, load_meta
from sprag.embed import JinaEmbedder
from sprag.assemble import ChunkPlacement, patched_full_attn, make_inv_freq_for
from sprag.mags.calibrate import (
    grab_last_residual, fit_mags, save_mags, DEFAULT_MAGS_LAYERS,
)


def find_needle_chunk(meta: dict, needle_text: str) -> int:
    """Find chunk_id whose text overlaps with the needle (substring)."""
    # We compare against a distinctive substring of the needle.
    key = needle_text.split(".")[0][:60].strip()
    for c in meta["chunks"]:
        if key.lower() in c["text_preview"].lower():
            return c["id"]
    # Fall back: the chunk containing the needle's first 20 chars
    for c in meta["chunks"]:
        if needle_text[:20].lower() in c["text_preview"].lower():
            return c["id"]
    return -1


def build_placement(cache_dir, chunk_ids, chunk_lookup, prefix_len: int):
    placements = []
    flat_ids = []
    cursor = prefix_len
    for cid in chunk_ids:
        tensors = load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))
        ids = tensors["input_ids"]
        L = int(ids.shape[0])
        cached = {li: (tensors[f"K_l{li}"], tensors[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(chunk_lookup[cid]["a_start"]),
            b_start=cursor, length=L, cached=cached
        ))
        flat_ids.extend(ids.tolist())
        cursor += L
    return placements, flat_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--n_wrong", type=int, default=3, help="Number of wrong chunks for T-")
    ap.add_argument("--n_calib", type=int, default=30, help="Number of cases to use")
    ap.add_argument("--k_svd", type=int, default=4)
    args = ap.parse_args()

    cases = [json.loads(l) for l in args.cases.open()][: args.n_calib]
    print(f"Calibrating MAGS on {len(cases)} cases")

    model, tok, _ = load_model()
    emb = JinaEmbedder()
    inv_freq = make_inv_freq_for(model)

    pos_bucket = {li: [] for li in DEFAULT_MAGS_LAYERS}
    neg_bucket = {li: [] for li in DEFAULT_MAGS_LAYERS}

    for i, case in enumerate(cases):
        tmp = args.out.parent / f"_calib_cache_{i}"
        tmp.mkdir(parents=True, exist_ok=True)

        # Rebuild needle text from the (case data + filler context)
        needle_template_match = case["haystack"]
        # We don't carry needle text directly; reconstruct via answer picks
        picks = case["answer_picks"]
        needle_keywords = list(picks.values())  # e.g. ["Helena", "scarlet"]

        build_chunk_cache(model, tok, case["haystack"], tmp,
                          chunk_size=args.chunk_size, embed_fn=emb.encode_passage)
        meta = load_meta(tmp)
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        # Find oracle chunk(s) by checking which chunks contain ALL needle keywords
        oracle_ids = []
        for c in meta["chunks"]:
            text = c["text_preview"]
            # text_preview is only first 120 chars; load the actual chunk text via tokens
            full = tok.decode(load_file(str(tmp / f"chunk_{c['id']:05d}.safetensors"))["input_ids"])
            if all(kw.lower() in full.lower() for kw in needle_keywords):
                oracle_ids.append(c["id"])
        if not oracle_ids:
            print(f"  case {i}: no oracle chunk (needle straddles); skipping")
            continue

        # Bottom-K chunks (most-dissimilar from query) for T-
        q_vec = emb.encode_query(["Q: " + case["question"] + "\nA:"])[0]
        all_repr = torch.stack([load_file(str(tmp / f"chunk_{c['id']:05d}.safetensors"))["chunk_repr"]
                                 for c in meta["chunks"]], 0).float()
        q = torch.nn.functional.normalize(q_vec.unsqueeze(0), dim=-1)
        c = torch.nn.functional.normalize(all_repr, dim=-1)
        sims = (q @ c.T).squeeze(0)
        sorted_idx = torch.argsort(sims).tolist()
        wrong_ids = [meta["chunks"][k]["id"] for k in sorted_idx if meta["chunks"][k]["id"] not in oracle_ids][: args.n_wrong]
        print(f"  case {i}: oracle={oracle_ids[:2]} wrong={wrong_ids}")

        prompt_tail_ids = tok("Q: " + case["question"] + "\nA:", add_special_tokens=False).input_ids

        # T+: oracle placement
        pos_placements, pos_flat = build_placement(tmp, oracle_ids[:1], chunk_lookup, prefix_len=0)
        pos_input = torch.tensor([pos_flat + prompt_tail_ids], dtype=torch.long)
        with torch.no_grad(), patched_full_attn(model, pos_placements, inv_freq=inv_freq), \
                grab_last_residual(model) as cap:
            model.model(pos_input, use_cache=False)
        for li in DEFAULT_MAGS_LAYERS:
            pos_bucket[li].append(cap[li])

        # T-: wrong-chunk placement
        neg_placements, neg_flat = build_placement(tmp, wrong_ids, chunk_lookup, prefix_len=0)
        neg_input = torch.tensor([neg_flat + prompt_tail_ids], dtype=torch.long)
        with torch.no_grad(), patched_full_attn(model, neg_placements, inv_freq=inv_freq), \
                grab_last_residual(model) as cap:
            model.model(neg_input, use_cache=False)
        for li in DEFAULT_MAGS_LAYERS:
            neg_bucket[li].append(cap[li])

    pos = {li: torch.stack(v, 0) for li, v in pos_bucket.items() if v}
    neg = {li: torch.stack(v, 0) for li, v in neg_bucket.items() if v}
    print(f"Collected pos={list(pos.values())[0].shape}  neg={list(neg.values())[0].shape}")

    params = fit_mags(pos, neg, k=args.k_svd, tau_quantile=0.95)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_mags(params, args.out)
    print(f"Saved MAGS params to {args.out}")
    for li in params.layer_indices:
        print(f"  layer {li}: B shape={tuple(params.B[li].shape)}  tau={params.tau[li]:.3f}")


if __name__ == "__main__":
    main()
