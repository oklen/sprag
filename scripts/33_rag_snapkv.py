"""RAG-faithful (precomputable) SnapKV — anchor-as-observation vs question-as-obs.

Motivation: in a real RAG system the chunk KV-caches must be precomputed BEFORE the
query is known, so the SnapKV observation window CANNOT be the question (that is only
available at query time). Faithful design (variant B):
  - anchor = retrieval top-1 chunk = our ORACLE chunk (the one containing the gold
    answer). Placed LAST in the context, kept in FULL (uncompressed prefix/anchor).
  - all OTHER chunks (the "compress region") are SnapKV-compressed, scored by the
    ANCHOR's queries (query-independent → precomputable).
  - at query time the question is appended and decoded.

Per record we measure three arms on the SAME layout (anchor-last), compressing the
SAME region (non-anchor) and keeping anchor+question in all of them — so the only
difference between A and B is the scoring query:
  - FRESH : full context, no compression (baseline).
  - B     : compress non-anchor region scored by ANCHOR queries  (precomputable).
  - A     : compress non-anchor region scored by QUESTION queries (upper bound, NOT
            precomputable) — gives the "precompute cost" gap A − B.

Metric: greedy-gen alias ACC. Same clean chat pipeline as scripts/32 (A3B-Instruct,
chat template + EOS). No RoPE shift (SnapKV keeps post-RoPE keys at original pos).

Env: SPRAG_MODEL_PATH (default /tmp/Qwen3-30B-A3B-Instruct-2507).
Modes: sanity (identity gate A & B) | coverage (the experiment).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
_USER_PREFIX = "<|im_start|>user\n"


def load_model(model_path=None):
    model_path = model_path or os.environ.get(
        "SPRAG_MODEL_PATH", "/tmp/Qwen3-30B-A3B-Instruct-2507")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True)
    model.eval()
    return model, tok


def emb_device(model):
    return model.get_input_embeddings().weight.device


@contextlib.contextmanager
def eager_attn(model):
    saved, cfgs = {}, {id(model.config): model.config}
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None:
            cfgs[id(c)] = c
    for c in cfgs.values():
        saved[id(c)] = getattr(c, "_attn_implementation", None)
        with contextlib.suppress(Exception):
            c._attn_implementation = "eager"
    try:
        yield
    finally:
        for c in cfgs.values():
            with contextlib.suppress(Exception):
                c._attn_implementation = saved[id(c)] or "sdpa"


def _layer_kv(cache, li):
    if hasattr(cache, "layers"):
        return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


def _num_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def _slice_cache(cache, n):
    new = DynamicCache()
    for li in range(_num_layers(cache)):
        K, V = _layer_kv(cache, li)
        new.update(K[:, :, :n, :].contiguous(), V[:, :, :n, :].contiguous(), li)
    return new


def strip_think(s):
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.S)
    s = re.sub(r"<think>.*", " ", s, flags=re.S)
    return s.strip()


def alias_match(pred, answers):
    p = strip_think(pred).lower()
    return any(a and a.lower() in p for a in answers)


def text_has_alias(tok, ids_slice, answers):
    t = tok.decode(ids_slice, skip_special_tokens=True).lower()
    return any(a and a.lower() in t for a in answers)


def _eos_ids(tok):
    s = {tok.eos_token_id}
    with contextlib.suppress(Exception):
        s.add(tok.convert_tokens_to_ids("<|im_end|>"))
    return {x for x in s if x is not None and x >= 0}


def _cfg(model):
    c = getattr(model.config, "text_config", model.config)
    return c.num_attention_heads, c.num_key_value_heads


def chunk_positions(n, cs):
    return [(s, min(s + cs, n)) for s in range(0, n, cs)]


def q_suffix_ids(tok, query):
    return tok("\n\nQuestion: " + query + "<|im_end|>\n<|im_start|>assistant\n",
               add_special_tokens=False).input_ids


# ---- SnapKV core (region-scoped) ---------------------------------------- #

def select_indices(attn_tuple, region, ratio, kernel, n_q, n_kv):
    """Per-layer per-kvhead Top-K indices over the FIRST `region` key positions
    (the compress region). attn[li]: [1,Qh,q_len,kv_len]. Returns (idxs, Kc)."""
    Kc = max(1, min(region, round(ratio * region)))
    grp = n_q // n_kv
    idxs = []
    for a in attn_tuple:
        s = a[..., :region].float().mean(dim=2)        # [1,Qh,region]
        s = s.view(1, n_kv, grp, region).mean(dim=2)   # [1,n_kv,region]
        if kernel > 1 and region > kernel:
            s = F.max_pool1d(s, kernel_size=kernel, stride=1, padding=kernel // 2)
            s = s[..., :region]
        idx = s.topk(Kc, dim=-1).indices
        idx, _ = torch.sort(idx, dim=-1)
        idxs.append(idx)
    return idxs, Kc


def build_comp_region(cache_full, idxs, L_comp, L_anchor):
    """Compressed = per-head gathered compress-region KV (Kc) + FULL anchor KV."""
    new = DynamicCache()
    for li in range(_num_layers(cache_full)):
        K, V = _layer_kv(cache_full, li)               # [1,n_kv,L_comp+L_anchor(+..),d]
        d = K.shape[-1]
        gi = idxs[li].to(K.device).unsqueeze(-1).expand(-1, -1, -1, d)
        Kc = torch.gather(K[:, :, :L_comp, :], 2, gi)
        Vc = torch.gather(V[:, :, :L_comp, :], 2, gi)
        Ka = K[:, :, L_comp:L_comp + L_anchor, :]
        Va = V[:, :, L_comp:L_comp + L_anchor, :]
        new.update(torch.cat([Kc, Ka], 2).contiguous(),
                   torch.cat([Vc, Va], 2).contiguous(), li)
    return new


def decode_q(model, tok, cache, q_ids, real_start, max_new, device, eos):
    """Prefill q_ids at absolute positions [real_start..] over `cache`, then greedy."""
    out = []
    cur = cache
    sl = cur.get_seq_length()
    with torch.no_grad():
        o = model(input_ids=torch.tensor([q_ids], device=device),
                  position_ids=torch.tensor([list(range(real_start, real_start + len(q_ids)))], device=device),
                  past_key_values=cur,
                  cache_position=torch.arange(sl, sl + len(q_ids), device=device), use_cache=True)
        cur = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        pos = real_start + len(q_ids)
        for _ in range(max_new):
            if nxt in eos:
                break
            out.append(nxt); sl = cur.get_seq_length()
            o = model(input_ids=torch.tensor([[nxt]], device=device),
                      position_ids=torch.tensor([[pos]], device=device),
                      past_key_values=cur,
                      cache_position=torch.tensor([sl], device=device), use_cache=True)
            cur = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            pos += 1
    return tok.decode(out, skip_special_tokens=True)


def build_layout(tok, ctx, answers, max_ctx, chunk_size):
    """Return (prefix_ids, anchor_ids) with anchor = first chunk containing the gold
    alias (oracle top-1), moved to the END; prefix = user-header + non-anchor chunks.
    Returns None if no anchor chunk or too few chunks."""
    doc = tok(ctx, add_special_tokens=False).input_ids[:max_ctx]
    chunks = chunk_positions(len(doc), chunk_size)
    if len(chunks) < 2:
        return None
    t = -1
    for ci, (s, e) in enumerate(chunks):
        if text_has_alias(tok, doc[s:e], answers):
            t = ci; break
    if t < 0:
        return None
    anchor = doc[chunks[t][0]:chunks[t][1]]
    non = [doc[p] for ci, (s, e) in enumerate(chunks) if ci != t for p in range(s, e)]
    pre = tok(_USER_PREFIX, add_special_tokens=False).input_ids
    prefix = pre + non
    if len(prefix) < 64:
        return None
    return prefix, anchor


# ---- modes --------------------------------------------------------------- #

def _score_and_caches(model, tok, prefix_ids, anchor_ids, q_ids, device, n_q, n_kv):
    """One prefill of compress-region, eager anchor forward (B scores), eager q
    forward (A scores). Returns (cache_full[L_ctx], attn_anchor, attn_q, L_comp, L_anchor)."""
    L_comp = len(prefix_ids); L_anchor = len(anchor_ids); L_ctx = L_comp + L_anchor
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([prefix_ids], device=device),
                      use_cache=True).past_key_values          # L_comp
        with eager_attn(model):
            oa = model(input_ids=torch.tensor([anchor_ids], device=device),
                       past_key_values=cache,
                       cache_position=torch.arange(L_comp, L_ctx, device=device),
                       position_ids=torch.tensor([list(range(L_comp, L_ctx))], device=device),
                       output_attentions=True, use_cache=True)
            cache_full = oa.past_key_values                    # L_ctx
            attn_anchor = oa.attentions                        # [.,Qh,L_anchor,L_ctx]
            q_probe = _slice_cache(cache_full, L_ctx)
            oq = model(input_ids=torch.tensor([q_ids], device=device),
                       past_key_values=q_probe,
                       cache_position=torch.arange(L_ctx, L_ctx + len(q_ids), device=device),
                       position_ids=torch.tensor([list(range(L_ctx, L_ctx + len(q_ids)))], device=device),
                       output_attentions=True, use_cache=True)
            attn_q = oq.attentions                             # [.,Qh,W_q,L_ctx+W_q]
    return cache_full, attn_anchor, attn_q, L_comp, L_anchor


def mode_sanity(model, tok, args):
    device = emb_device(model)
    n_q, n_kv = _cfg(model)
    eos = _eos_ids(tok)
    print(f"heads q={n_q} kv={n_kv}")
    ctx = ("The Eiffel Tower is in Paris. " * 40 +
           "The secret password is ZEBRA-42. " +
           "Bananas are yellow and grow in bunches. " * 40)
    ans = ["ZEBRA-42", "ZEBRA"]
    q = "What is the secret password?"
    lay = build_layout(tok, ctx, ans, args.max_ctx, args.chunk_size)
    if lay is None:
        print("  sanity: no anchor chunk (unexpected)"); return
    prefix, anchor = lay
    q_ids = q_suffix_ids(tok, q)
    cache_full, attn_a, attn_q, L_comp, L_anchor = _score_and_caches(
        model, tok, prefix, anchor, q_ids, device, n_q, n_kv)
    L_ctx = L_comp + L_anchor
    print(f"L_comp={L_comp} L_anchor={L_anchor} L_ctx={L_ctx} "
          f"attn_a={None if not attn_a else tuple(attn_a[0].shape)} "
          f"attn_q={None if not attn_q else tuple(attn_q[0].shape)}")
    if not attn_a or not attn_q:
        print("  !! eager output_attentions failed"); return
    g_fresh = decode_q(model, tok, _slice_cache(cache_full, L_ctx), q_ids, L_ctx, args.max_new_tokens, device, eos)
    print(f"  FRESH: {g_fresh[:90]!r}  match={alias_match(g_fresh, ans)}")
    # identity gate: ratio=1 keep-all compress region, kernel=1 → both == fresh
    for tag, attn in (("B(anchor)", attn_a), ("A(question)", attn_q)):
        idxs, Kc = select_indices(attn, L_comp, 1.0, 1, n_q, n_kv)
        comp = build_comp_region(cache_full, idxs, L_comp, L_anchor)
        g = decode_q(model, tok, comp, q_ids, L_ctx, args.max_new_tokens, device, eos)
        print(f"  KEEP-ALL {tag} (Kc={Kc}): identical_to_fresh={g==g_fresh}")
    for r in (0.05, 0.1, 0.2):
        line = f"  r={r}:"
        for tag, attn in (("B", attn_a), ("A", attn_q)):
            idxs, Kc = select_indices(attn, L_comp, r, args.kernel, n_q, n_kv)
            comp = build_comp_region(cache_full, idxs, L_comp, L_anchor)
            g = decode_q(model, tok, comp, q_ids, L_ctx, args.max_new_tokens, device, eos)
            line += f"  {tag}(Kc={Kc})={int(alias_match(g, ans))}"
        print(line)


def load_longbench(path, limit=None):
    recs = []
    for line in open(path):
        recs.append(json.loads(line))
        if limit and len(recs) >= limit:
            break
    return recs


def _in_shard(args, gi):
    return (gi % args.num_shards) == args.shard_id


def mode_coverage(model, tok, args):
    device = emb_device(model)
    n_q, n_kv = _cfg(model)
    eos = _eos_ids(tok)
    ratios = args.ratios
    out = {"model": os.environ.get("SPRAG_MODEL_PATH"), "ratios": ratios,
           "kernel": args.kernel, "chunk_size": args.chunk_size,
           "design": "anchor=oracle top-1 chunk moved last, kept full; compress non-anchor; "
                     "B=anchor-obs(precomputable) A=question-obs(upper bound)",
           "metric": "alias_acc", "shard_id": args.shard_id,
           "num_shards": args.num_shards, "datasets": {}}
    done = set()
    if args.resume and args.out.exists():
        with contextlib.suppress(Exception):
            out = json.loads(args.out.read_text()); out.setdefault("datasets", {})
            for ds, rows in out["datasets"].items():
                for r in rows:
                    done.add((ds, r["case"]))
    print(f"[resume] {len(done)} done" if done else "[fresh start]")
    gi = -1
    for ds in args.data:
        rows = out["datasets"].setdefault(ds, [])
        for ri, rec in enumerate(load_longbench(args.bench / f"{ds}.jsonl", args.limit)):
            gi += 1
            if not _in_shard(args, gi) or (ds, ri) in done:
                continue
            ctx, q, ans = rec["context"], rec["input"], rec["answers"]
            lay = build_layout(tok, ctx, ans, args.max_ctx, args.chunk_size)
            if lay is None:
                continue
            prefix, anchor = lay
            q_ids = q_suffix_ids(tok, q)
            cache_full, attn_a, attn_q, L_comp, L_anchor = _score_and_caches(
                model, tok, prefix, anchor, q_ids, device, n_q, n_kv)
            if not attn_a or not attn_q:
                print("FATAL no attentions"); return
            L_ctx = L_comp + L_anchor
            g_fresh = decode_q(model, tok, _slice_cache(cache_full, L_ctx), q_ids, L_ctx, args.max_new_tokens, device, eos)
            mf = int(alias_match(g_fresh, ans))
            row = {"case": ri, "L_comp": L_comp, "L_anchor": L_anchor, "L_ctx": L_ctx, "acc_fresh": mf}
            for r in ratios:
                cell = {}
                for tag, attn in (("B", attn_a), ("A", attn_q)):
                    idxs, Kc = select_indices(attn, L_comp, r, args.kernel, n_q, n_kv)
                    comp = build_comp_region(cache_full, idxs, L_comp, L_anchor)
                    g = decode_q(model, tok, comp, q_ids, L_ctx, args.max_new_tokens, device, eos)
                    cell[f"acc_{tag}"] = int(alias_match(g, ans))
                    cell["Kc"] = Kc
                cell["keep_frac"] = round((cell["Kc"] + L_anchor) / L_ctx, 4)
                row[f"r{r}"] = cell
                print(f"  [{ds} {ri}] Lc={L_comp} La={L_anchor} fresh={mf} r={r} "
                      f"Kc={cell['Kc']} B={cell['acc_B']} A={cell['acc_A']}")
            del cache_full, attn_a, attn_q
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(row)
            tmp = args.out.with_suffix(".tmp"); tmp.write_text(json.dumps(out, indent=1))
            os.replace(tmp, args.out)
    _summary(out, ratios)
    print("wrote", args.out)


def _summary(out, ratios):
    allrows = [r for rows in out["datasets"].values() for r in rows]
    n = len(allrows)
    print(f"\n=== RAG-SnapKV (anchor=oracle top-1) pooled, n={n} ===")
    if not n:
        return
    af = sum(r["acc_fresh"] for r in allrows) / n
    print(f"acc_fresh(full ctx)={af:.3f}")
    print(f"{'ratio':>6} {'keepf':>7} {'acc_B':>6} {'ΔB':>7} {'acc_A':>6} {'ΔA':>7} {'gapA-B':>7}")
    for r in ratios:
        k = f"r{r}"
        c = [row[k] for row in allrows if k in row]
        if not c:
            continue
        b = sum(x["acc_B"] for x in c) / len(c)
        a = sum(x["acc_A"] for x in c) / len(c)
        kf = sum(x["keep_frac"] for x in c) / len(c)
        print(f"{r:6.2f} {kf:7.3f} {b:6.3f} {b-af:+7.3f} {a:6.3f} {a-af:+7.3f} {a-b:+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["sanity", "coverage"])
    ap.add_argument("--data", nargs="+", default=["2wikimqa", "hotpotqa", "musique"])
    ap.add_argument("--bench", type=Path, default=ROOT / "data/benchmarks/longbench_v1/data")
    ap.add_argument("--out", type=Path, default=Path("data/ragsnap_out.json"))
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--max_ctx", type=int, default=12000)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard_id", type=int, default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    print(f"model on {emb_device(model)}; type={model.config.model_type}")
    t0 = time.time()
    {"sanity": mode_sanity, "coverage": mode_coverage}[args.mode](model, tok, args)
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
