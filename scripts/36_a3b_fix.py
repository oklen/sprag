"""Big-model (full-attention) cached-vs-fresh coverage experiment — A3B main run.

Per docs/A100_EXPERIMENT_GUIDE.md §5.3 the splice is done at the HF cache level.
UNIFIED convention (to be directly comparable to the 27B hybrid run,
31_hybrid_coverage.py): COMPACTED assembly + shift_rope.

  - assembly = [sink ∪ ctx-chunks ∪ target] placed CONTIGUOUSLY (compact
    positions 0..L-1), then query+gold continuing.
  - fresh : the compact assembly recomputed from scratch (plain forward).
  - cached: each kept chunk's full-doc K/V (from one full-doc forward) is
    shift_rope'd from its ORIGINAL position to its COMPACT position and used as
    the prefix cache (prefill-skip); query+gold attend to it. A3B is qwen3_moe →
    EVERY layer's K/V is spliceable (48/48), so cached reuse can be lossless.
  - The ONLY difference vs the 27B run is how many layers are spliceable
    (A3B 48/48 vs 27B 16/64) → isolates the hybrid limitation.

Metric: gold-answer NLL/PPL (cached vs fresh) + greedy-gen alias accuracy
(<think> stripped for the reasoning model).

Modes: sanity (delta=0 splice == plain forward; PPL scorer) | gate (E1) |
coverage (E2). Env: SPRAG_MODEL_PATH (default /tmp/Qwen3-30B-A3B-Thinking-2507).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sprag.rope import build_inv_freq, shift_rope


def load_model(model_path=None):
    model_path = model_path or os.environ.get(
        "SPRAG_MODEL_PATH", "/tmp/Qwen3-30B-A3B-Thinking-2507")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True)
    model.eval()
    return model, tok


def emb_device(model):
    return model.get_input_embeddings().weight.device


def inv_freq_for(model):
    cfg = model.config
    tc = getattr(cfg, "text_config", cfg)
    rp = getattr(tc, "rope_parameters", None) or {}
    theta = rp.get("rope_theta", getattr(tc, "rope_theta", 1e7))
    prf = rp.get("partial_rotary_factor", getattr(tc, "partial_rotary_factor", 1.0))
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    return build_inv_freq(head_dim=hd, partial_rotary_factor=prf, rope_theta=theta)


# ---- DynamicCache helpers (version-defensive) ---------------------------- #

def _layer_kv(cache, li):
    if hasattr(cache, "layers"):
        return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


def _num_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def build_full_cache(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=True).past_key_values


def build_compacted_cache(full_cache, kept_ranges, inv_freq):
    """Compact the kept (a0,a1) ranges to contiguous positions, shift_rope each
    chunk's cached K from original→compact position. Returns (DynamicCache, L)."""
    offsets, b = [], 0
    for (a0, a1) in kept_ranges:
        offsets.append(b); b += a1 - a0
    new = DynamicCache()
    for li in range(_num_layers(full_cache)):
        K, V = _layer_kv(full_cache, li)
        ifq = inv_freq.to(K.device)
        Ks, Vs = [], []
        for (a0, a1), b0 in zip(kept_ranges, offsets):
            kc = K[:, :, a0:a1, :]
            delta = b0 - a0
            Ks.append(shift_rope(kc, delta, ifq) if delta != 0 else kc)
            Vs.append(V[:, :, a0:a1, :])
        new.update(torch.cat(Ks, 2).contiguous(), torch.cat(Vs, 2).contiguous(), li)
    return new, b


# ---- chunking / scoring -------------------------------------------------- #

def chunk_positions(n, cs):
    return [(s, min(s + cs, n)) for s in range(0, n, cs)]


def text_has_alias(tok, ids_slice, answers):
    t = tok.decode(ids_slice, skip_special_tokens=True).lower()
    return any(a and a.lower() in t for a in answers)


def alias_match(pred, answers):
    p = strip_think(pred).lower()
    return any(a and a.lower() in p for a in answers)


def strip_think(s):
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.S)
    s = re.sub(r"<think>.*", " ", s, flags=re.S)   # unterminated
    return s.strip()


CHAT = False
_USER_PREFIX = "<|im_start|>user\n"


def _prefix_ids(tok):
    return tok(_USER_PREFIX, add_special_tokens=False).input_ids if CHAT else []


def _q_g_ids(tok, query, gold):
    if CHAT:
        q = tok("\n\nQuestion: " + query + "<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False).input_ids
    else:
        q = tok("\n\nQuestion: " + query + "\nAnswer:", add_special_tokens=False).input_ids
    g = tok(" " + gold, add_special_tokens=False).input_ids
    return q, g


def _nll(logits, seq_len, g_ids):
    ng = len(g_ids)
    rows = logits[seq_len - ng - 1: seq_len - 1].float()
    lp = torch.log_softmax(rows, -1)
    return -sum(lp[j, t].item() for j, t in enumerate(g_ids)) / ng


def gold_nll_plain(model, tok, ids_list, pos_list, query, gold, device):
    """Fresh / no-splice: plain forward over ids_list (with pos_list) + q + g."""
    q_ids, g_ids = _q_g_ids(tok, query, gold)
    seq = list(ids_list) + q_ids + g_ids
    last = pos_list[-1] if pos_list else -1
    pos = list(pos_list) + list(range(last + 1, last + 1 + len(q_ids) + len(g_ids)))
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([seq], device=device),
                       position_ids=torch.tensor([pos], device=device),
                       use_cache=False).logits[0]
    return _nll(logits, len(seq), g_ids)


def gold_nll_cached(model, tok, sub, L, query, gold, device):
    q_ids, g_ids = _q_g_ids(tok, query, gold)
    new = q_ids + g_ids
    pos = list(range(L, L + len(new)))
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([new], device=device),
                       position_ids=torch.tensor([pos], device=device),
                       past_key_values=sub,
                       cache_position=torch.arange(L, L + len(new), device=device),
                       use_cache=True).logits[0]
    return _nll(logits, len(new), g_ids)


def _greedy(model, tok, prefix_ids, prefix_pos, past, max_new, device, answers=None, check_every=64):
    Lk = past.get_seq_length() if past is not None else 0
    _eos = {tok.eos_token_id}
    _ie = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(_ie, int) and _ie >= 0:
        _eos.add(_ie)
    out, last_pos, cur = [], prefix_pos[-1], past
    with torch.no_grad():
        o = model(input_ids=torch.tensor([prefix_ids], device=device),
                  position_ids=torch.tensor([prefix_pos], device=device),
                  past_key_values=cur,
                  cache_position=torch.arange(Lk, Lk + len(prefix_ids), device=device),
                  use_cache=True)
        cur = o.past_key_values; nxt = int(o.logits[0, -1].argmax())
        for _ in range(max_new):
            if nxt in _eos:
                break
            out.append(nxt); last_pos += 1; sl = cur.get_seq_length()
            if answers and len(out) % check_every == 0 and alias_match(tok.decode(out, skip_special_tokens=True), answers):
                break
            o = model(input_ids=torch.tensor([[nxt]], device=device),
                      position_ids=torch.tensor([[last_pos]], device=device),
                      past_key_values=cur,
                      cache_position=torch.tensor([sl], device=device), use_cache=True)
            cur = o.past_key_values; nxt = int(o.logits[0, -1].argmax())
    return tok.decode(out, skip_special_tokens=True)


def gen_plain(model, tok, ids_list, pos_list, query, device, max_new, answers=None):
    q_ids, _ = _q_g_ids(tok, query, "")
    seq = list(ids_list) + q_ids
    last = pos_list[-1] if pos_list else -1
    pos = list(pos_list) + list(range(last + 1, last + 1 + len(q_ids)))
    return _greedy(model, tok, seq, pos, None, max_new, device, answers=answers)


def gen_cached(model, tok, sub, L, query, device, max_new, answers=None):
    q_ids, _ = _q_g_ids(tok, query, "")
    return _greedy(model, tok, q_ids, list(range(L, L + len(q_ids))), sub, max_new, device, answers=answers)


def load_longbench(path, limit=None):
    recs = []
    for line in open(path):
        recs.append(json.loads(line))
        if limit and len(recs) >= limit:
            break
    return recs


# ---- modes --------------------------------------------------------------- #

def mode_sanity(model, tok, args):
    device = emb_device(model)
    print("=== SANITY 1: keep-all (delta=0) cached splice ≈ plain forward ===")
    doc = "The Eiffel Tower is in Paris. " * 60 + "The secret code is ZEBRA-42. " + \
          "Bananas are yellow. " * 60
    ids = tok(doc, return_tensors="pt").input_ids.to(device)
    N = ids.shape[1]
    full = build_full_cache(model, ids)
    inv_freq = inv_freq_for(model)
    sub, L = build_compacted_cache(full, [(0, N)], inv_freq)   # delta 0 → verbatim
    probe = tok(" The secret code is", add_special_tokens=False).input_ids
    with torch.no_grad():
        lc = model(input_ids=torch.tensor([probe], device=device),
                   position_ids=torch.tensor([list(range(L, L + len(probe)))], device=device),
                   past_key_values=sub,
                   cache_position=torch.arange(L, L + len(probe), device=device),
                   use_cache=True).logits[0]
        catted = torch.cat([ids[0], torch.tensor(probe, device=device)])[None]
        lf = model(input_ids=catted, use_cache=False).logits[0][N: N + len(probe)]
    d = (lc.float() - lf.float()).abs().max().item()
    rel = d / lf.float().abs().max().item()
    print(f"  max|Δlogit|={d:.4e} rel={rel:.4e} → {'PASS' if rel < 5e-2 else 'FAIL'}")
    print("=== SANITY 2: gold-NLL separates gold from distractor ===")
    cids = tok("Maria's favorite color is purple.", add_special_tokens=False).input_ids
    for ans in ["purple", "green", "banana"]:
        print(f"  NLL({ans!r})={gold_nll_plain(model, tok, cids, list(range(len(cids))), 'What is Maria favorite color?', ans, device):.3f}")


def _in_shard(args, gi):
    """Stride record-sharding for multi-process concurrency. gi = global record
    index over all (dataset, record) pairs iterated so far."""
    return (gi % args.num_shards) == args.shard_id


def _done_gi(path):
    """gi's already recorded in a save_gen JSONL (for --resume across reclaims)."""
    done = set()
    if path and os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)["gi"])
            except Exception:
                pass
    return done


def mode_gate(model, tok, args):
    device = emb_device(model)
    res = {}
    done = _done_gi(args.save_gen) if (args.save_gen and args.resume) else set()
    if done:
        print(f"[resume] skipping {len(done)} already-saved records")
    gi = -1
    for ds in args.data:
        recs = load_longbench(args.bench / f"{ds}.jsonl", args.limit)
        fn = nn_ = 0.0; fo = no = lo = n = 0
        for rec in recs:
            gi += 1
            if not _in_shard(args, gi):
                continue
            if gi in done:
                continue
            ctx, q, ans = rec["context"], rec["input"], rec["answers"]; gold = ans[0]
            cids = _prefix_ids(tok) + tok(ctx, add_special_tokens=False).input_ids[: args.max_ctx]
            _pre = _prefix_ids(tok)
            nf = gold_nll_plain(model, tok, cids, list(range(len(cids))), q, gold, device)
            n0 = gold_nll_plain(model, tok, _pre, list(range(len(_pre))), q, gold, device)
            fn += nf; nn_ += n0; n += 1; lo += int(nf < n0)
            g_full = gen_plain(model, tok, cids, list(range(len(cids))), q, device, args.max_new_tokens, answers=(ans if args.early_exit_match else None))
            g_no = gen_plain(model, tok, _pre, list(range(len(_pre))), q, device, args.max_new_tokens, answers=(ans if args.early_exit_match else None))
            mf = int(alias_match(g_full, ans)); mn = int(alias_match(g_no, ans))
            fo += mf; no += mn
            if args.save_gen:
                rec_out = {"ds": ds, "gi": gi, "q": q, "gold": gold, "answers": ans,
                           "ppl_full": pow(2.71828, nf), "ppl_no": pow(2.71828, n0),
                           "match_full": mf, "match_no": mn,
                           "think_closed_full": int("</think>" in g_full),
                           "think_closed_no": int("</think>" in g_no),
                           "pred_full": strip_think(g_full)[:400], "pred_no": strip_think(g_no)[:400],
                           "gen_full": g_full, "gen_no": g_no}
                with open(args.save_gen, "a") as gf:
                    gf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(f"  [{ds} {n}] PPL full={pow(2.71828, nf):.2f} no={pow(2.71828, n0):.2f} "
                  f"match f/n={mf}/{mn} thinkclosed={int('</think>' in g_full)} genlen={len(g_full)}")
        res[ds] = {"n": n, "ppl_full": pow(2.71828, fn / n), "ppl_no": pow(2.71828, nn_ / n),
                   "acc_full": fo / n, "acc_no": no / n, "pct_ctx_lowers_nll": lo / n}
        print(f"=== {ds}: {res[ds]} ==="); args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


def mode_coverage(model, tok, args):
    device = emb_device(model)
    inv_freq = inv_freq_for(model)
    covs = args.coverages
    out = {"model": os.environ.get("SPRAG_MODEL_PATH"), "chunk_size": args.chunk_size,
           "coverages": covs, "convention": "compacted+shift_rope",
           "shard_id": args.shard_id, "num_shards": args.num_shards, "datasets": {}}
    done = {}
    if args.resume and args.out.exists():
        try:
            out = json.loads(args.out.read_text())
            out["shard_id"] = args.shard_id; out["num_shards"] = args.num_shards
            out.setdefault("datasets", {})
            for _ds, _rows in out["datasets"].items():
                done[_ds] = {r["case"] for r in _rows}
            print(f"[resume] loaded {sum(len(v) for v in done.values())} done coverage records")
        except Exception as _e:
            print(f"[resume] load failed ({_e}); starting fresh")
            out["datasets"] = {}; done = {}
    gi = -1
    for ds in args.data:
        rows = out["datasets"].get(ds, [])
        for ri, rec in enumerate(load_longbench(args.bench / f"{ds}.jsonl", args.limit)):
            gi += 1
            if not _in_shard(args, gi):
                continue
            if ri in done.get(ds, set()):
                continue
            ctx, q, ans = rec["context"], rec["input"], rec["answers"]; gold = ans[0]
            doc_ids = _prefix_ids(tok) + tok(ctx, add_special_tokens=False).input_ids[: args.max_ctx]
            chunks = chunk_positions(len(doc_ids), args.chunk_size)
            t = -1
            for ci, (s, e) in enumerate(chunks):
                if ci >= args.min_depth and text_has_alias(tok, doc_ids[s:e], ans):
                    t = ci; break
            if t < 0:
                continue
            full = build_full_cache(model, torch.tensor([doc_ids], device=device))
            row = {"case": ri, "target": t, "n_prec": t}
            for c in covs:
                inc = round(c / 100.0 * t)
                ctx_idx = list(range(t - inc, t))
                contam = any(text_has_alias(tok, doc_ids[chunks[i][0]:chunks[i][1]], ans) for i in ctx_idx)
                # sink-dup fix: only prepend the explicit sink when chunk0 is NOT
                # already kept (else doc[0:M] is duplicated → handicaps fresh decode)
                _sink = [] if (0 in ctx_idx) else [(0, args.M)]
                kept = _sink + [chunks[i] for i in ctx_idx] + [chunks[t]]
                comp_ids = [doc_ids[p] for (a0, a1) in kept for p in range(a0, a1)]
                sub, L = build_compacted_cache(full, kept, inv_freq)
                nll_c = gold_nll_cached(model, tok, sub, L, q, gold, device)
                nll_f = gold_nll_plain(model, tok, comp_ids, list(range(L)), q, gold, device)
                pc = alias_match(gen_cached(model, tok, *build_compacted_cache(full, kept, inv_freq), q, device, args.max_new_tokens), ans)
                pf = alias_match(gen_plain(model, tok, comp_ids, list(range(L)), q, device, args.max_new_tokens), ans)
                cell = {"ppl_cached": pow(2.71828, nll_c), "ppl_fresh": pow(2.71828, nll_f),
                        "acc_cached": int(pc), "acc_fresh": int(pf),
                        "contam": int(contam), "ntok": L}
                row[f"c{int(c)}"] = cell
                print(f"  [{ds} {ri}] t={t} cov={int(c):3d}% {'CONTAM' if contam else 'clean'} "
                      f"PPL c={cell['ppl_cached']:.2f}/f={cell['ppl_fresh']:.2f} "
                      f"acc c={int(pc)}/f={int(pf)} ntok={L}")
            del full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(row); out["datasets"][ds] = rows
            _tmp = args.out.with_suffix(args.out.suffix + ".tmp")
            _tmp.write_text(json.dumps(out, indent=1)); os.replace(str(_tmp), str(args.out))
    _summary(out, covs)
    print("wrote", args.out)


def _summary(out, covs):
    print("\n=== E2 coverage summary (pooled, clean cells) ===")
    print(f"{'cov%':>5s} {'n':>5s} {'PPL_f':>8s} {'ΔPPL':>8s} {'acc_f':>6s} {'Δacc':>6s}")
    for c in covs:
        k = f"c{int(c)}"
        cells = [r[k] for rows in out["datasets"].values() for r in rows if k in r and not r[k]["contam"]]
        if not cells:
            continue
        n = len(cells)
        print(f"{int(c):5d} {n:5d} {sum(x['ppl_fresh'] for x in cells)/n:8.2f} "
              f"{sum(x['ppl_cached']-x['ppl_fresh'] for x in cells)/n:+8.2f} "
              f"{sum(x['acc_fresh'] for x in cells)/n:6.2f} "
              f"{sum(x['acc_cached']-x['acc_fresh'] for x in cells)/n:+6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["sanity", "gate", "coverage"])
    ap.add_argument("--data", nargs="+", default=["2wikimqa", "hotpotqa", "musique"])
    ap.add_argument("--bench", type=Path, default=ROOT / "data/benchmarks/longbench_v1/data")
    ap.add_argument("--out", type=Path, default=Path("data/big_out.json"))
    ap.add_argument("--coverages", type=float, nargs="+", default=[0, 25, 50, 75, 100])
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--min_depth", type=int, default=2)
    ap.add_argument("--max_ctx", type=int, default=12000)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard_id", type=int, default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
    ap.add_argument("--save_gen", type=str, default=None,
                    help="append raw generations (gate mode) to this JSONL for spot-checking")
    ap.add_argument("--early_exit_match", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip records already present in --save_gen (survive worker reclaims)")
    args = ap.parse_args()
    globals()["CHAT"] = args.chat
    if args.save_gen:
        Path(args.save_gen).parent.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            open(args.save_gen, "w").close()  # truncate at start (per-shard file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    print(f"model on {emb_device(model)}; n_layers={model.config.num_hidden_layers}; type={model.config.model_type}")
    t0 = time.time()
    {"sanity": mode_sanity, "gate": mode_gate, "coverage": mode_coverage}[args.mode](model, tok, args)
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
