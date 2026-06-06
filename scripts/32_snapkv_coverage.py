"""SnapKV (attention-importance KV compression) vs Fresh — A3B-Instruct, chat mode.

Compares SnapKV-compressed KV-cache decoding against full (fresh) decoding on the
SAME clean chat pipeline used for the coverage experiment (chat template + EOS,
non-thinking Instruct model → untruncated outputs).

SnapKV (Li et al. 2024) mechanism, implemented faithfully:
  - prompt = [<|im_start|>user\n  +  doc(context, the COMPRESS region)]  +  q_suffix
             (question + assistant header = the OBSERVATION WINDOW, kept in full).
  - profile: observation-window queries attend to context keys; per-(kv)head
    importance = mean over obs rows of softmax(Q_obs·K_ctx^T).
  - 1D max-pool (kernel k) over the score sequence → contiguous clusters (vs H2O).
  - per-head Top-K context tokens kept (K = ratio · L_ctx), + full obs window.
  - decode from the compressed cache. NO RoPE shift: kept keys keep their original
    positions (post-RoPE K stored verbatim) → relative distances exact.

Crucial perf detail: PREFILL stays SDPA (eager would materialise an O(L^2) attn
matrix over the long context). Only the SHORT obs-window forward (q_len = W) is run
under eager + output_attentions to harvest the scores — tiny memory.

Metric: greedy-gen alias accuracy (ACC), SnapKV vs Fresh, across kept ratios.
Budget-invariance / no-truncation already established for this pipeline.

Env: SPRAG_MODEL_PATH (default /tmp/Qwen3-30B-A3B-Instruct-2507).
Modes: sanity (identity gate + smoke) | coverage (the experiment).
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
    """Temporarily flip attn_implementation to eager so output_attentions works.
    Used ONLY for the short obs-window forward; prefill stays sdpa."""
    saved = {}
    cfgs = {id(model.config): model.config}
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None:
            cfgs[id(c)] = c
    for c in cfgs.values():
        saved[id(c)] = getattr(c, "_attn_implementation", None)
        try:
            c._attn_implementation = "eager"
        except Exception:
            pass
    try:
        yield
    finally:
        for c in cfgs.values():
            with contextlib.suppress(Exception):
                c._attn_implementation = saved[id(c)] if saved[id(c)] is not None else "sdpa"


# ---- DynamicCache helpers (version-defensive) ---------------------------- #

def _layer_kv(cache, li):
    if hasattr(cache, "layers"):
        return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


def _num_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def _slice_cache(cache, n):
    """New cache keeping the first n positions of every layer (drops the rest)."""
    new = DynamicCache()
    for li in range(_num_layers(cache)):
        K, V = _layer_kv(cache, li)
        new.update(K[:, :, :n, :].contiguous(), V[:, :, :n, :].contiguous(), li)
    return new


# ---- text / chat helpers ------------------------------------------------- #

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


def build_prompt_ids(tok, ctx, query, max_ctx):
    """Return (prefix_ids, obs_ids): prefix = user-header + context (COMPRESS
    region); obs = question + assistant header (kept in full, the scoring window)."""
    pre = tok(_USER_PREFIX, add_special_tokens=False).input_ids
    doc = tok(ctx, add_special_tokens=False).input_ids[:max_ctx]
    prefix = pre + doc
    obs = tok("\n\nQuestion: " + query + "<|im_end|>\n<|im_start|>assistant\n",
              add_special_tokens=False).input_ids
    return prefix, obs


# ---- SnapKV core --------------------------------------------------------- #

def snapkv_scores(model, prefix_ids, obs_ids, device):
    """Prefill prefix (sdpa) then score with one eager obs-window forward.
    Returns (full_cache[L_ctx+W], attn_tuple, L_ctx, W). attn[li]: [1,Qh,W,L_ctx+W]."""
    L_ctx = len(prefix_ids)
    W = len(obs_ids)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([prefix_ids], device=device),
                      use_cache=True).past_key_values
        with eager_attn(model):
            out = model(input_ids=torch.tensor([obs_ids], device=device),
                        past_key_values=cache,
                        cache_position=torch.arange(L_ctx, L_ctx + W, device=device),
                        position_ids=torch.tensor([list(range(L_ctx, L_ctx + W))], device=device),
                        output_attentions=True, use_cache=True)
    return out.past_key_values, out.attentions, L_ctx, W


def select_indices(attn_tuple, L_ctx, W, ratio, kernel, n_q, n_kv):
    """Per-layer, per-kvhead Top-K context indices (pooled). Returns list[li]->idx
    tensor [1,n_kv,Kc]. Kc = round(ratio*L_ctx), clamped to [1,L_ctx]."""
    Kc = max(1, min(L_ctx, round(ratio * L_ctx)))
    grp = n_q // n_kv
    idxs = []
    for a in attn_tuple:
        s = a[..., :L_ctx].float().mean(dim=2)            # [1,Qh,L_ctx]
        s = s.view(1, n_kv, grp, L_ctx).mean(dim=2)        # [1,n_kv,L_ctx] (GQA group)
        if kernel > 1 and L_ctx > kernel:
            s = F.max_pool1d(s, kernel_size=kernel, stride=1, padding=kernel // 2)
            s = s[..., :L_ctx]
        idx = s.topk(Kc, dim=-1).indices                   # [1,n_kv,Kc]
        idx, _ = torch.sort(idx, dim=-1)
        idxs.append(idx)
    return idxs, Kc


def build_compressed(full_cache, idxs, L_ctx, W):
    """Compressed = per-head gathered context KV  +  obs KV minus its LAST token
    (last obs token is fed at decode time to produce the first answer token)."""
    new = DynamicCache()
    for li in range(_num_layers(full_cache)):
        K, V = _layer_kv(full_cache, li)                   # [1,n_kv,L_ctx+W,d]
        d = K.shape[-1]
        idx = idxs[li].to(K.device)                        # [1,n_kv,Kc]
        gi = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        Kc = torch.gather(K[:, :, :L_ctx, :], 2, gi)
        Vc = torch.gather(V[:, :, :L_ctx, :], 2, gi)
        Ko = K[:, :, L_ctx:L_ctx + W - 1, :]               # obs minus last token
        Vo = V[:, :, L_ctx:L_ctx + W - 1, :]
        new.update(torch.cat([Kc, Ko], 2).contiguous(),
                   torch.cat([Vc, Vo], 2).contiguous(), li)
    return new


def decode_from(model, tok, cache, last_id, real_pos, max_new, device, eos):
    """Greedy decode starting by feeding `last_id` at absolute position real_pos
    (cache currently holds get_seq_length() entries). Returns decoded text."""
    out = []
    cur = cache
    pos = real_pos
    with torch.no_grad():
        sl = cur.get_seq_length()
        o = model(input_ids=torch.tensor([[last_id]], device=device),
                  position_ids=torch.tensor([[pos]], device=device),
                  past_key_values=cur,
                  cache_position=torch.tensor([sl], device=device), use_cache=True)
        cur = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        for _ in range(max_new):
            if nxt in eos:
                break
            out.append(nxt); pos += 1; sl = cur.get_seq_length()
            o = model(input_ids=torch.tensor([[nxt]], device=device),
                      position_ids=torch.tensor([[pos]], device=device),
                      past_key_values=cur,
                      cache_position=torch.tensor([sl], device=device), use_cache=True)
            cur = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
    return tok.decode(out, skip_special_tokens=True)


# ---- modes --------------------------------------------------------------- #

def _cfg(model):
    c = getattr(model.config, "text_config", model.config)
    return c.num_attention_heads, c.num_key_value_heads


def mode_sanity(model, tok, args):
    device = emb_device(model)
    n_q, n_kv = _cfg(model)
    eos = _eos_ids(tok)
    print(f"heads: q={n_q} kv={n_kv} group={n_q//n_kv}")
    doc = ("The Eiffel Tower is in Paris. " * 40 +
           "The secret password is ZEBRA-42. " +
           "Bananas are yellow and grow in bunches. " * 40)
    q = "What is the secret password?"
    prefix, obs = build_prompt_ids(tok, doc, q, args.max_ctx)
    full, attn, L_ctx, W = snapkv_scores(model, prefix, obs, device)
    print(f"L_ctx={L_ctx} W={W} attn_layers={None if attn is None else len(attn)} "
          f"attn0_shape={None if not attn else tuple(attn[0].shape)}")
    if not attn:
        print("  !! output_attentions returned None — eager toggle FAILED, need hook fallback")
        return
    real_pos = L_ctx + W - 1
    last_id = obs[-1]
    # Fresh decode (full cache minus last obs token)
    fresh_cache = _slice_cache(full, L_ctx + W - 1)
    g_fresh = decode_from(model, tok, fresh_cache, last_id, real_pos, args.max_new_tokens, device, eos)
    print(f"  FRESH: {g_fresh[:120]!r}")
    # Identity gate: ratio=1.0, no pooling → keep all context → should match fresh
    idxs, Kc = select_indices(attn, L_ctx, W, 1.0, 1, n_q, n_kv)
    comp = build_compressed(full, idxs, L_ctx, W)
    g_id = decode_from(model, tok, comp, last_id, real_pos, args.max_new_tokens, device, eos)
    print(f"  KEEP-ALL (Kc={Kc}): {g_id[:120]!r}  identical={g_id==g_fresh}")
    # Low-ratio smoke
    for r in (0.1, 0.2):
        # rebuild scores cache each time (build_compressed consumes obs slice only; full intact)
        idxs, Kc = select_indices(attn, L_ctx, W, r, args.kernel, n_q, n_kv)
        comp = build_compressed(full, idxs, L_ctx, W)
        g = decode_from(model, tok, comp, last_id, real_pos, args.max_new_tokens, device, eos)
        print(f"  SnapKV r={r} (Kc={Kc}): {g[:120]!r}  match={alias_match(g, ['ZEBRA-42','ZEBRA'])}")


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
    done = set()
    if args.resume and args.out.exists():
        with contextlib.suppress(Exception):
            prev = json.loads(args.out.read_text())
            for ds, rows in prev.get("datasets", {}).items():
                for r in rows:
                    done.add((ds, r["case"]))
    out = {"model": os.environ.get("SPRAG_MODEL_PATH"), "ratios": ratios,
           "kernel": args.kernel, "obs_window": "q_suffix", "metric": "alias_acc",
           "shard_id": args.shard_id, "num_shards": args.num_shards, "datasets": {}}
    if args.resume and args.out.exists():
        out = json.loads(args.out.read_text())
        out.setdefault("datasets", {})
    print(f"[resume] {len(done)} records already done" if done else "[fresh start]")
    gi = -1
    for ds in args.data:
        rows = out["datasets"].setdefault(ds, [])
        for ri, rec in enumerate(load_longbench(args.bench / f"{ds}.jsonl", args.limit)):
            gi += 1
            if not _in_shard(args, gi) or (ds, ri) in done:
                continue
            ctx, q, ans = rec["context"], rec["input"], rec["answers"]
            prefix, obs = build_prompt_ids(tok, ctx, q, args.max_ctx)
            if len(prefix) < 64:
                continue
            full, attn, L_ctx, W = snapkv_scores(model, prefix, obs, device)
            if not attn:
                print("FATAL: no attentions"); return
            real_pos = L_ctx + W - 1
            last_id = obs[-1]
            fresh_cache = _slice_cache(full, L_ctx + W - 1)
            g_fresh = decode_from(model, tok, fresh_cache, last_id, real_pos, args.max_new_tokens, device, eos)
            mf = int(alias_match(g_fresh, ans))
            row = {"case": ri, "L_ctx": L_ctx, "W": W, "acc_fresh": mf}
            for r in ratios:
                idxs, Kc = select_indices(attn, L_ctx, W, r, args.kernel, n_q, n_kv)
                comp = build_compressed(full, idxs, L_ctx, W)
                g = decode_from(model, tok, comp, last_id, real_pos, args.max_new_tokens, device, eos)
                mc = int(alias_match(g, ans))
                row[f"r{r}"] = {"Kc": Kc, "acc": mc,
                                "keep_frac": round((Kc + W) / (L_ctx + W), 4)}
                print(f"  [{ds} {ri}] L_ctx={L_ctx} fresh={mf} r={r} Kc={Kc} acc={mc}")
            del full, attn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(row)
            tmp = args.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, indent=1))
            os.replace(tmp, args.out)
    _summary(out, ratios)
    print("wrote", args.out)


def _summary(out, ratios):
    print("\n=== SnapKV vs Fresh (pooled ACC) ===")
    allrows = [r for rows in out["datasets"].values() for r in rows]
    n = len(allrows)
    if not n:
        return
    af = sum(r["acc_fresh"] for r in allrows) / n
    print(f"n={n}  acc_fresh={af:.3f}")
    print(f"{'ratio':>6s} {'keepfrac':>9s} {'acc_snap':>9s} {'Δvs_fresh':>10s}")
    for r in ratios:
        k = f"r{r}"
        cells = [row[k] for row in allrows if k in row]
        if not cells:
            continue
        acc = sum(c["acc"] for c in cells) / len(cells)
        kf = sum(c["keep_frac"] for c in cells) / len(cells)
        print(f"{r:6.2f} {kf:9.3f} {acc:9.3f} {acc-af:+10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["sanity", "coverage"])
    ap.add_argument("--data", nargs="+", default=["2wikimqa", "hotpotqa", "musique"])
    ap.add_argument("--bench", type=Path, default=ROOT / "data/benchmarks/longbench_v1/data")
    ap.add_argument("--out", type=Path, default=Path("data/snapkv_out.json"))
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--kernel", type=int, default=7)
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
