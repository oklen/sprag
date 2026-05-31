"""§5ad — phantom-context frame probe.

Use the FULL cache (chunk K/V built in [anchor][real preceding doc][chunk] — the
chunk's hidden states already encode the real context; the 'phantom context' is
baked in). At USE time put the chunk back at its BUILD position (M + a_start) so
its cached K need ZERO RoPE shift (geometry identical to build), and vary ONLY
what fills the frame positions [M, M+a_start) that the query reads across:

  real  — the actual preceding doc tokens  (reconstructs the exact build prefix
          → fresh chunk K/V == the cache; this arm is the full-reprefill UPPER
          BOUND and a faithfulness sanity: splice == no-op here).
  ph    — a repeated placeholder token (content-free, precomputable): "just
          occupy positions / release attention".
  rand  — random token ids (content present but WRONG, non-fluent gibberish):
          does the model need *any* frame content, or the *right* content?
  rother— a DIFFERENT document's leading tokens, same length (fluent but
          irrelevant): does the frame need the chunk's OWN context, or does any
          fluent text suffice? (rother≈rand → needs right content; rother≈real
          → fluency alone suffices.)
  shift — no frame; chunk spliced at M the standard way (cached K shifted by
          -a_start). Reference = what plain full-cache splicing gives today.

Question (user, 2026-05-31): given the chunk KV already encodes real context,
does the model just need *a* frame to release/absorb attention (ph == real), or
must it extract real semantic content from the frame (ph << real)? Single gold
chunk (k=1) removes the sibling-splice confound.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import capture_full_attn_kv, split_into_chunks
from sprag.assemble import ChunkPlacement, patched_full_attn, make_inv_freq_for

_spec = importlib.util.spec_from_file_location("mk12", ROOT / "scripts" / "12_sink_mk.py")
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)
_spec2 = importlib.util.spec_from_file_location("d18", ROOT / "scripts" / "18_delta_cache.py")
d18 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(d18)


def build_full_kv(model, tokens, anchor_ids, device):
    """One [anchor][doc] forward; return per-layer full K/V (n_kv, M+N, hd) and M.
    Chunk cid's cache = K[:, M+a0:M+a1] built at absolute position M+a0."""
    M = len(anchor_ids)
    full_ids = torch.tensor([anchor_ids + tokens.tolist()], dtype=torch.long, device=device)
    with torch.no_grad(), capture_full_attn_kv(model) as kv:
        model(input_ids=full_ids, use_cache=False)
    K = {li: kv[li]["K"][0].clone() for li in FULL_ATTN_LAYERS}
    V = {li: kv[li]["V"][0].clone() for li in FULL_ATTN_LAYERS}
    return K, V, M


def run_arm(model, tok, device, inv_freq, sink_ids, frame_ids, ctoks, canon, cached,
            question, max_new):
    """Assemble [sink][frame][chunk][query], splice the chunk's full cache.

    The chunk lands at b_start = M + len(frame); the cache (built at canon) is
    RoPE-shifted by (b_start - canon). A length-`a_start` frame restores the
    build position (delta 0); an empty frame = standard splice at M (delta
    -a_start); a short frame of length W = standard-ish splice at M+W (the chunk
    is position-SHIFTED, not restored)."""
    flat = list(sink_ids) + list(frame_ids) + list(ctoks)
    b_start = len(sink_ids) + len(frame_ids)
    pl = ChunkPlacement(a_start=canon, b_start=b_start, length=len(ctoks), cached=cached)
    tail = tok("\n\nQ: " + question + "\nA:", add_special_tokens=False).input_ids
    inp = torch.tensor([flat + tail], dtype=torch.long, device=device)
    with torch.no_grad(), patched_full_attn(model, [pl], inv_freq=inv_freq, alpha=1.0):
        out = model.generate(input_ids=inp, max_new_tokens=max_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=ROOT / "data/mk/suite_8k")
    ap.add_argument("--out", type=Path, default=ROOT / "data/phantom_frame_mk.json")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--ph_token", default="<|endoftext|>",
                    help="placeholder token string for the 'ph' arm")
    ap.add_argument("--arms", nargs="+",
                    default=["real", "rother", "ph", "shift", "fsent",
                             "sflu16", "sflu64", "sflu256", "sph64"])
    ap.add_argument("--max_a0", type=int, default=None,
                    help="skip golds whose build position a_start exceeds this "
                         "(caps frame-forward length / cost)")
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
    ph_tok = tok.convert_tokens_to_ids(args.ph_token)
    rng = torch.Generator().manual_seed(0)
    vocab = model.config.vocab_size

    # fluent-but-irrelevant donor pool for the 'rother' arm: a different case's
    # haystack tokens. case0 borrows from the last case; others borrow case0.
    def _toks(ci):
        return tok((args.suite / f"case_{ci:02d}" / "haystack.txt").read_text(),
                   return_tensors="pt").input_ids[0]
    donor0 = _toks(case_ids[0])
    donor_last = _toks(case_ids[-1])

    # fixed fluent sentence, tiled to any length (exp2 'fsent'): repetitive but
    # in-distribution — sits between ph (repeated token) and rother (coherent).
    fsent_unit = tok("The history of science is a long and winding road, full of "
                     "unexpected discoveries and quiet, patient work. ",
                     add_special_tokens=False).input_ids

    cnt = {a: 0 for a in args.arms}
    n = 0
    rows = []

    for ci in case_ids:
        cd = args.suite / f"case_{ci:02d}"
        haystack = (cd / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd / "queries.jsonl").open()]
        tokens = tok(haystack, return_tensors="pt").input_ids[0]
        chunks = split_into_chunks(tokens, chunk_size=args.chunk_size)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        K, V, M = build_full_kv(model, tokens, anchor_ids, device)
        print(f"case{ci}: {len(chunks)} chunks, full cache built")

        for q in queries:
            gold = d18.find_gold(chunks, tok, tokens,
                                 mk.reconstruct_needle(q["template_id"], q["picks"]))
            if gold < 0:
                continue
            ch = chunk_by_id[gold]
            a0, a1 = ch.a_start, ch.a_end
            if args.max_a0 is not None and a0 > args.max_a0:
                continue
            canon = M + a0
            L = a1 - a0
            ctoks = tokens[a0:a1].tolist()
            cached = {li: (K[li][:, canon:M + a1, :].contiguous(),
                           V[li][:, canon:M + a1, :].contiguous()) for li in FULL_ATTN_LAYERS}

            donor = donor_last if ci == case_ids[0] else donor0
            tiled = (fsent_unit * (a0 // len(fsent_unit) + 1))[:a0]
            frames = {
                # full-length, position-restored (delta 0)
                "real": tokens[:a0].tolist(),
                "rother": donor[:a0].tolist(),
                "ph": [ph_tok] * a0,
                "rand": torch.randint(1000, vocab - 1000, (a0,), generator=rng).tolist(),
                "fsent": tiled,
                "shift": [],
                # short frame, chunk position-SHIFTED to M+W (exp1)
                "sflu16": donor[:16].tolist(),
                "sflu64": donor[:64].tolist(),
                "sflu256": donor[:256].tolist(),
                "sph64": [ph_tok] * 64,
            }
            n += 1
            row = {"case": ci, "id": q["id"], "gold": gold, "a0": a0}
            line = f"  c{ci} q{q['id']} a0={a0}"
            for arm in args.arms:
                t0 = time.time()
                out = run_arm(model, tok, device, inv_freq, anchor_ids, frames[arm],
                              ctoks, canon, cached, q["question"], args.max_new_tokens)
                cls = mk.classify(out, q["answer"], q["distractor_answers"])
                cnt[arm] += cls == "correct"
                row[arm] = {"output": out, "class": cls, "time": time.time() - t0}
                line += f"  {arm}={'Y' if cls == 'correct' else '.'}"
            print(line)
            rows.append(row)

    print(f"\n=== phantom-frame probe (k=1 gold, M={args.M}, ph={args.ph_token!r}, n={n}) ===")
    for arm in args.arms:
        print(f"  {arm:6s}: {cnt[arm]}/{n}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"M": args.M, "ph_token": args.ph_token, "arms": args.arms,
         "counts": cnt, "n": n, "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
