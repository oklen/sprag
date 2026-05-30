"""§5z — Blend the LINEAR-attn (GatedDeltaNet) recurrent state.

Everything in sprag so far splices only the 6 full-attn layers; the 18
GatedDeltaNet layers are always recomputed fresh. This probe caches each
chunk's from-zero recurrent fold and, at assembly, blends it into the
end-of-prefill state the decode reads:

    S_used = alpha * S_cached + (1 - alpha) * S_fresh   (per linear layer)

S_cached for a retrieval set = SUM of the per-chunk from-zero folds (the linear
state has no position-independent per-chunk slice — it's a gated sequential
fold — so a chunk's isolated fold is the only cacheable unit; sum is the
crudest composition). alpha=0 reproduces the raw (linear-fresh) path EXACTLY,
which is the built-in sanity check.

Assembly mirrors raw_oracle_k3 from 12_sink_mk: full attention is left FRESH
(no K/V splice) so any movement is attributable to the linear state alone.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import torch

from sprag.loader import load_model, LINEAR_ATTN_LAYERS
from sprag.chunk_cache import build_chunk_cache, load_meta
from sprag.embed import JinaEmbedder
from sprag.assemble import patched_linear_state, compute_chunk_linear_states

# Reuse the MK helpers (module name starts with a digit -> importlib).
_spec = importlib.util.spec_from_file_location("mk12", ROOT / "scripts" / "12_sink_mk.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--M", type=int, default=4, help="sink length (first M doc tokens)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--norm_match", action="store_true",
                    help="Rescale composed cached state to the fresh state's "
                         "per-head norm (isolate direction from scale).")
    ap.add_argument("--limit_cases", type=int, default=None)
    ap.add_argument("--reuse_cache", action="store_true")
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    emb = JinaEmbedder()

    alpha_keys = [f"a{a:g}" for a in args.alphas]
    counts = {k: {"correct": 0, "distractor": 0, "other": 0} for k in alpha_keys}
    n_q = 0
    rows = []

    for ci in case_ids:
        cd_src = args.suite / f"case_{ci:02d}"
        haystack = (cd_src / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd_src / "queries.jsonl").open()]
        cache_dir = args.suite / f"_lin_case{ci:02d}"

        if not (args.reuse_cache and cache_dir.exists()):
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            build_chunk_cache(model, tok, haystack, cache_dir,
                              chunk_size=args.chunk_size, embed_fn=emb.encode_passage)
        meta = load_meta(cache_dir)

        sink_ids = mk._load_chunk(cache_dir, 0)["input_ids"][: args.M].tolist()
        lin_state_cache: dict[int, dict] = {}  # chunk_id -> {li: S}

        def chunk_state(cid):
            if cid not in lin_state_cache:
                ids = mk._load_chunk(cache_dir, cid)["input_ids"]
                lin_state_cache[cid] = compute_chunk_linear_states(model, ids)
            return lin_state_cache[cid]

        for q in queries:
            gold_needle = mk.reconstruct_needle(q["template_id"], q["picks"])
            gold_chunk = mk.find_chunk_for_needle(cache_dir, meta, tok, gold_needle)
            if gold_chunk < 0:
                print(f"  case{ci} q{q['id']} SKIP: no gold chunk")
                continue
            siblings = []
            for qo in queries:
                if qo["id"] == q["id"]:
                    continue
                cid = mk.find_chunk_for_needle(
                    cache_dir, meta, tok,
                    mk.reconstruct_needle(qo["template_id"], qo["picks"]))
                if 0 <= cid != gold_chunk and cid not in siblings:
                    siblings.append(cid)
            ids_k3 = [gold_chunk] + siblings[:2]

            # Assembled raw prefix: sink + gold + 2 siblings (full attn FRESH).
            flat = list(sink_ids)
            for cid in ids_k3:
                flat.extend(mk._load_chunk(cache_dir, cid)["input_ids"].tolist())
            tail = tok("\n\nQ: " + q["question"] + "\nA:",
                       add_special_tokens=False).input_ids
            inp = torch.tensor([flat + tail], dtype=torch.long, device=device)

            # Composed cached linear state = SUM over the 3 retrieved chunks.
            composed = {li: None for li in LINEAR_ATTN_LAYERS}
            for cid in ids_k3:
                st = chunk_state(cid)
                for li in LINEAR_ATTN_LAYERS:
                    s = st[li]
                    composed[li] = s.clone() if composed[li] is None else composed[li] + s

            n_q += 1
            row = {"case": ci, "id": q["id"], "ids_k3": ids_k3}
            for a, akey in zip(args.alphas, alpha_keys):
                t0 = time.time()
                with torch.no_grad(), patched_linear_state(
                        model, composed, alpha=a, norm_match=args.norm_match):
                    out_ids = model.generate(
                        input_ids=inp, max_new_tokens=args.max_new_tokens,
                        do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
                out = tok.decode(out_ids[0, inp.shape[1]:], skip_special_tokens=True)
                dt = time.time() - t0
                cls = mk.classify(out, q["answer"], q["distractor_answers"])
                counts[akey][cls] += 1
                row[akey] = {"output": out, "class": cls, "time": dt}
                print(f"  case{ci} q{q['id']} a={a:<4g} {dt:4.1f}s "
                      f"[{cls:10s}] {out[:50]!r}")
            rows.append(row)

    summary = {k: {"correct": counts[k]["correct"], "n": n_q,
                   "acc": counts[k]["correct"] / max(1, n_q)} for k in alpha_keys}
    print("\n=== linear-state blend (full-attn FRESH), n =", n_q, "===")
    for a, k in zip(args.alphas, alpha_keys):
        c = counts[k]
        print(f"  alpha={a:<4g}  {c['correct']:3d}/{n_q}  "
              f"(dist {c['distractor']}, other {c['other']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"M": args.M, "chunk_size": args.chunk_size, "alphas": args.alphas,
         "summary": summary, "counts": counts, "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
