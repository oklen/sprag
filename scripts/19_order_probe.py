"""Premise check for the linear-recency idea: does CHUNK ORDER affect accuracy?

Same oracle k=3 retrieval set, plain raw assembly (full attn + linear both
fresh), but the gold chunk placed FIRST vs LAST (closest to Q). A large
gold-last > gold-first gap = strong recency bias (the RNN-like linear fold +
RoPE favouring the most-recent chunk). If order barely matters, the
chunk-last-ensemble idea won't help.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from sprag.loader import load_model
from sprag.chunk_cache import split_into_chunks
_spec = importlib.util.spec_from_file_location("mk12", ROOT / "scripts" / "12_sink_mk.py")
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)
_spec2 = importlib.util.spec_from_file_location("d18", ROOT / "scripts" / "18_delta_cache.py")
d18 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(d18)


def gen(model, tok, device, sink_ids, chunk_tok_lists, question, max_new):
    flat = list(sink_ids)
    for c in chunk_tok_lists:
        flat.extend(c)
    tail = tok("\n\nQ: " + question + "\nA:", add_special_tokens=False).input_ids
    inp = torch.tensor([flat + tail], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(input_ids=inp, max_new_tokens=max_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=ROOT / "data/mk/suite_8k")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[:args.limit_cases]
    model, tok, _ = load_model()
    device = next(model.parameters()).device
    anchor = tok.convert_tokens_to_ids("<|endoftext|>")

    orders = ["gold_first", "gold_last", "gold_mid"]
    cnt = {o: 0 for o in orders}
    n = 0
    for ci in case_ids:
        cd = args.suite / f"case_{ci:02d}"
        haystack = (cd / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd / "queries.jsonl").open()]
        tokens = tok(haystack, return_tensors="pt").input_ids[0]
        chunks = split_into_chunks(tokens, chunk_size=args.chunk_size)
        toks = {c.chunk_id: tokens[c.a_start:c.a_end].tolist() for c in chunks}
        sink_ids = tokens[:args.M].tolist()
        for q in queries:
            gold = d18.find_gold(chunks, tok, tokens,
                                 mk.reconstruct_needle(q["template_id"], q["picks"]))
            if gold < 0:
                continue
            sib = []
            for qo in queries:
                if qo["id"] == q["id"]:
                    continue
                c = d18.find_gold(chunks, tok, tokens,
                                  mk.reconstruct_needle(qo["template_id"], qo["picks"]))
                if 0 <= c != gold and c not in sib:
                    sib.append(c)
            sib = sib[:2]
            arr = {
                "gold_first": [gold] + sib,
                "gold_last": sib + [gold],
                "gold_mid": ([sib[0], gold] + sib[1:]) if len(sib) == 2 else [gold] + sib,
            }
            n += 1
            line = f"  c{ci} q{q['id']}"
            for o in orders:
                out = gen(model, tok, device, sink_ids, [toks[c] for c in arr[o]],
                          q["question"], args.max_new_tokens)
                ok = mk.classify(out, q["answer"], q["distractor_answers"]) == "correct"
                cnt[o] += ok
                line += f"  {o}={'Y' if ok else '.'}"
            print(line)
    print(f"\n=== order sensitivity (oracle k=3, n={n}) ===")
    for o in orders:
        print(f"  {o:11s}: {cnt[o]}/{n}")


if __name__ == "__main__":
    main()
