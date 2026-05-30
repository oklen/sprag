"""§5aa — full−pos+fresh delta cache.

Idea (user, 2026-05-30): for each chunk, build two caches —
  full = chunk K/V in [anchor][real preceding context][chunk]   (sees real ctx)
  pos  = chunk K/V in [anchor]⟨position gap⟩[chunk]             (same abs pos,
         anchor kept, context tokens erased)
The residual (full − pos) isolates the *real context's content contribution*
with the position/anchor baseline subtracted out. At assembly we shift that
residual to the chunk's new position and ADD it onto the fresh K/V:
    K = fresh + α·shift_rope(full_K − pos_K)
    V = fresh + α·(full_V − pos_V)
Intent: keep the cross-context "memory" the chunk got from its original
document context, but rebase position/local-context to the new assembly.
α=0 reproduces pure fresh (raw) exactly — the sanity baseline.

This file = the K/V target. Linear-state and both-families targets follow.
Assembly mirrors raw_oracle_k3: sink (M anchor tokens, fresh) + gold + 2
siblings; full attention is fresh except for the added residual.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import torch

from sprag.loader import load_model, FULL_ATTN_LAYERS, LINEAR_ATTN_LAYERS
from sprag.chunk_cache import capture_full_attn_kv, split_into_chunks
from sprag.embed import JinaEmbedder  # noqa: F401 (kept for parity / future retrieval)
from sprag.assemble import (DeltaPlacement, patched_full_attn_delta, make_inv_freq_for,
                            patched_linear_state, compute_chunk_linear_states,
                            compute_running_linear_states,
                            ChunkPlacement, patched_full_attn)

_spec = importlib.util.spec_from_file_location("mk12", ROOT / "scripts" / "12_sink_mk.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)


def build_delta_kv(model, tok, tokens, chunks, anchor_ids, device, pos_fill="gap"):
    """Return delta[cid][li] = (dK, dV) and canon_pos[cid] for every chunk.
    dK is post-RoPE at its build position (M + a_start).

    pos_fill controls how the `pos` cache erases the real context while holding
    the chunk's absolute position:
      "gap"    — [anchor]+[chunk] with a position-id gap (chunk attends to only
                 the M anchor tokens; few keys).
      "anchor" — [anchor][anchor_tok × a_start][chunk] with natural positions
                 (chunk attends to M+a_start placeholder keys, matching `full`'s
                 key COUNT/position; differs from full only in context CONTENT).
    """
    M = len(anchor_ids)
    fill_tok = anchor_ids[0]
    doc = tokens.tolist()
    # FULL pass: [anchor][doc] in one forward; chunk i lands at M + a_start.
    full_ids = torch.tensor([anchor_ids + doc], dtype=torch.long, device=device)
    with torch.no_grad(), capture_full_attn_kv(model) as kv_full:
        model(input_ids=full_ids, use_cache=False)
    full_K = {li: kv_full[li]["K"][0] for li in FULL_ATTN_LAYERS}  # (n_kv, M+N, hd)
    full_V = {li: kv_full[li]["V"][0] for li in FULL_ATTN_LAYERS}

    delta, full, canon = {}, {}, {}
    for ch in chunks:
        a0, a1 = ch.a_start, ch.a_end
        cp = M + a0                                   # build position of chunk
        L = a1 - a0
        # POS pass: erase real context, keep chunk's absolute position + anchor.
        if pos_fill == "anchor":
            ids = torch.tensor([anchor_ids + [fill_tok] * a0 + doc[a0:a1]],
                               dtype=torch.long, device=device)
            with torch.no_grad(), capture_full_attn_kv(model) as kv_pos:
                model(input_ids=ids, use_cache=False)   # natural positions
            pos_slice = slice(-L, None)
        else:  # gap
            ids = torch.tensor([anchor_ids + doc[a0:a1]], dtype=torch.long, device=device)
            pos_ids = torch.tensor([list(range(M)) + list(range(cp, M + a1))],
                                   dtype=torch.long, device=device)
            with torch.no_grad(), capture_full_attn_kv(model) as kv_pos:
                model(input_ids=ids, position_ids=pos_ids, use_cache=False)
            pos_slice = slice(M, None)
        d, f = {}, {}
        for li in FULL_ATTN_LAYERS:
            fK = full_K[li][:, cp:M + a1, :].contiguous()
            fV = full_V[li][:, cp:M + a1, :].contiguous()
            pK = kv_pos[li]["K"][0][:, pos_slice, :]
            pV = kv_pos[li]["V"][0][:, pos_slice, :]
            d[li] = ((fK - pK).contiguous(), (fV - pV).contiguous())
            f[li] = (fK, fV)              # full cache for the `replace` (pos_new) mode
        delta[ch.chunk_id] = d
        full[ch.chunk_id] = f
        canon[ch.chunk_id] = cp
    return delta, full, canon


def build_delta_linear(model, tok, tokens, chunks, anchor_ids, device):
    """Return delta_S[cid][li] = full_S − pos_S for every chunk's recurrent state.
    full_S = running fold [anchor][doc] snapshot at the chunk boundary (real
    context); pos_S = from-zero fold of [anchor][chunk] (context erased)."""
    doc = tokens.tolist()
    seg_lists = [doc[ch.a_start:ch.a_end] for ch in chunks]
    full_S = compute_running_linear_states(model, anchor_ids, seg_lists)  # list per chunk
    delta = {}
    for i, ch in enumerate(chunks):
        pos_S = compute_chunk_linear_states(model, anchor_ids + seg_lists[i])
        delta[ch.chunk_id] = {li: full_S[i][li] - pos_S[li] for li in LINEAR_ATTN_LAYERS}
    return delta


def find_gold(chunks, tok, tokens, needle_text):
    # spine (middle 50 chars) is authoritative — it carries the city/street that
    # disambiguates same-template needles; head is only a fallback. (Mirrors
    # mk.find_chunk_for_needle: two separate passes, spine before head.)
    spine = needle_text[max(0, len(needle_text) // 2 - 25): len(needle_text) // 2 + 25].lower()
    texts = {ch.chunk_id: tok.decode(tokens[ch.a_start:ch.a_end]).lower() for ch in chunks}
    for ch in chunks:
        if spine in texts[ch.chunk_id]:
            return ch.chunk_id
    head = needle_text[:30].lower()
    for ch in chunks:
        if head in texts[ch.chunk_id]:
            return ch.chunk_id
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--M", type=int, default=4, help="anchor length (fixed token)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--target", choices=["kv", "linear", "both"], default="kv",
                    help="which family the full−pos+fresh delta is applied to")
    ap.add_argument("--mode", choices=["add", "replace"], default="add",
                    help="add = full−pos+fresh (residual ADDED onto fresh, §5aa); "
                         "replace = pos_new+(full−pos) = shift(full) REPLACING fresh "
                         "(the coherent variant; reduces to a standard cached splice). "
                         "replace is kv-only.")
    ap.add_argument("--strip", type=int, default=0,
                    help="(replace mode) drop the first S tokens of each chunk — the "
                         "most drift-prone boundary tokens (§5w sink_oracle_k3 used S=4).")
    ap.add_argument("--sink_doclead", action="store_true",
                    help="(replace mode) use the doc's first M tokens as the sink "
                         "instead of the eot anchor (§5w setup).")
    ap.add_argument("--pos_fill", choices=["gap", "anchor"], default="gap",
                    help="how the pos cache erases context: gap (position-id gap, "
                         "few keys) or anchor (fill context slots with anchor "
                         "placeholders → matches full's key count/position).")
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model)
    anchor_tok = tok.convert_tokens_to_ids("<|endoftext|>")
    anchor_ids = [anchor_tok] * args.M

    akeys = [f"a{a:g}" for a in args.alphas]
    counts = {k: {"correct": 0, "distractor": 0, "other": 0} for k in akeys}
    n_q = 0
    rows = []

    for ci in case_ids:
        cd = args.suite / f"case_{ci:02d}"
        haystack = (cd / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd / "queries.jsonl").open()]
        tokens = tok(haystack, return_tensors="pt").input_ids[0]
        chunks = split_into_chunks(tokens, chunk_size=args.chunk_size)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        delta = full_kv = canon = delta_lin = None
        if args.target in ("kv", "both"):
            delta, full_kv, canon = build_delta_kv(model, tok, tokens, chunks, anchor_ids,
                                                   device, pos_fill=args.pos_fill)
        if args.target in ("linear", "both"):
            delta_lin = build_delta_linear(model, tok, tokens, chunks, anchor_ids, device)
        print(f"case{ci}: {len(chunks)} chunks, delta built ({args.target})")

        for q in queries:
            gold = find_gold(chunks, tok, tokens,
                             mk.reconstruct_needle(q["template_id"], q["picks"]))
            if gold < 0:
                print(f"  case{ci} q{q['id']} SKIP no gold")
                continue
            sib = []
            for qo in queries:
                if qo["id"] == q["id"]:
                    continue
                c = find_gold(chunks, tok, tokens,
                              mk.reconstruct_needle(qo["template_id"], qo["picks"]))
                if 0 <= c != gold and c not in sib:
                    sib.append(c)
            ids_k3 = [gold] + sib[:2]

            sink_ids = tokens[:args.M].tolist() if args.sink_doclead else list(anchor_ids)
            flat = list(sink_ids)
            placements, cursor = [], args.M
            for cid in ids_k3:
                ch = chunk_by_id[cid]
                ctoks = tokens[ch.a_start:ch.a_end].tolist()
                if args.target in ("kv", "both"):
                    if args.mode == "replace":
                        S = args.strip
                        ctoks = ctoks[S:]                 # drop drift-prone head
                        cached = {li: (fK[:, S:, :].contiguous(), fV[:, S:, :].contiguous())
                                  for li, (fK, fV) in full_kv[cid].items()}
                        placements.append(ChunkPlacement(
                            a_start=canon[cid] + S, b_start=cursor, length=len(ctoks),
                            cached=cached))
                    else:
                        placements.append(DeltaPlacement(
                            b_start=cursor, length=len(ctoks),
                            canon_pos=canon[cid], delta=delta[cid]))
                flat.extend(ctoks)
                cursor += len(ctoks)
            tail = tok("\n\nQ: " + q["question"] + "\nA:", add_special_tokens=False).input_ids
            inp = torch.tensor([flat + tail], dtype=torch.long, device=device)

            # composed linear residual = SUM over retrieved chunks
            composed = None
            if args.target in ("linear", "both"):
                composed = {li: None for li in LINEAR_ATTN_LAYERS}
                for cid in ids_k3:
                    for li in LINEAR_ATTN_LAYERS:
                        s = delta_lin[cid][li]
                        composed[li] = s.clone() if composed[li] is None else composed[li] + s

            n_q += 1
            row = {"case": ci, "id": q["id"], "ids_k3": ids_k3}
            for a, ak in zip(args.alphas, akeys):
                t0 = time.time()
                with torch.no_grad(), contextlib.ExitStack() as stack:
                    if args.target in ("kv", "both"):
                        if args.mode == "replace":
                            stack.enter_context(patched_full_attn(
                                model, placements, inv_freq=inv_freq, alpha=a))
                        else:
                            stack.enter_context(patched_full_attn_delta(
                                model, placements, inv_freq=inv_freq, alpha=a))
                    if args.target in ("linear", "both"):
                        stack.enter_context(patched_linear_state(
                            model, composed, alpha=a, additive=True))
                    out_ids = model.generate(
                        input_ids=inp, max_new_tokens=args.max_new_tokens,
                        do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
                out = tok.decode(out_ids[0, inp.shape[1]:], skip_special_tokens=True)
                cls = mk.classify(out, q["answer"], q["distractor_answers"])
                counts[ak][cls] += 1
                row[ak] = {"output": out, "class": cls, "time": time.time() - t0}
                print(f"  case{ci} q{q['id']} a={a:<4g} [{cls:10s}] {out[:48]!r}")
            rows.append(row)

    print(f"\n=== {args.target} delta (full−pos+fresh), n={n_q} ===")
    for a, k in zip(args.alphas, akeys):
        print(f"  alpha={a:<4g}  {counts[k]['correct']:3d}/{n_q}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"target": args.target, "M": args.M, "alphas": args.alphas,
         "counts": counts, "n": n_q, "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
