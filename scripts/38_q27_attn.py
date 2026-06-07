"""Check #4: sink-duplication attention dilution on Qwen3.5-27B (full-attn layers).
For each c100 record build the DUP assembly ([sink]+chunks+target, chunk0 dups
doc[0:M]) and the NODUP assembly (no explicit sink). Capture the LAST query
position's attention (post-RoPE, GQA-expanded) over keys in the 16 full-attn
layers; record mass on [0:M) (sink copy1), [M:2M) (chunk0 head = sink copy2 in
dup / content in nodup), pos0, and total. Mean over heads+layers+records.
Tests: do the two sink copies hog attention (dup) and is it released (nodup)?
"""
import os, sys, json, argparse
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import split_into_chunks
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb, repeat_kv, eager_attention_forward, ALL_ATTENTION_FUNCTIONS)

CAP = {}  # layer_idx -> [m0M, mM2M, mpos0, nheads]

def make_capture(attn, M):
    def forward(hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kw):
        cfg = attn
        ish = hidden_states.shape[:-1]; hsh = (*ish, -1, cfg.head_dim)
        q, gate = torch.chunk(cfg.q_proj(hidden_states).view(*ish, -1, cfg.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*ish, -1)
        q = cfg.q_norm(q.view(hsh)).transpose(1, 2)
        k = cfg.k_norm(cfg.k_proj(hidden_states).view(hsh)).transpose(1, 2)
        v = cfg.v_proj(hidden_states).view(hsh).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if q.shape[-2] > 1:  # prefill -> capture last-row attention
            nrep = q.shape[1] // k.shape[1]
            kk = repeat_kv(k, nrep)
            scores = (q[:, :, -1:, :] @ kk.transpose(-2, -1)) * cfg.scaling  # (B,H,1,L)
            a = torch.softmax(scores.float(), dim=-1)[0, :, 0, :]  # (H,L)
            H = a.shape[0]
            m0M = a[:, :M].sum(-1).mean().item()
            mM2M = a[:, M:2 * M].sum(-1).mean().item()
            mp0 = a[:, 0].mean().item()
            c = CAP.setdefault(cfg.layer_idx, [0.0, 0.0, 0.0, 0])
            c[0] += m0M; c[1] += mM2M; c[2] += mp0; c[3] += 1
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, cfg.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
        ai = ALL_ATTENTION_FUNCTIONS.get_interface(cfg.config._attn_implementation, eager_attention_forward)
        ao, _ = ai(cfg, q, k, v, attention_mask, dropout=0.0, scaling=cfg.scaling, **kw)
        ao = ao.reshape(*ish, -1).contiguous() * torch.sigmoid(gate)
        return cfg.o_proj(ao), None
    return forward

def patch(model, M):
    orig = {}
    for li in FULL_ATTN_LAYERS:
        attn = model.model.layers[li].self_attn
        orig[li] = attn.forward
        attn.forward = make_capture(attn, M)
    return orig
def unpatch(model, orig):
    for li, fn in orig.items():
        model.model.layers[li].self_attn.forward = fn

def _prefix_ids(tok):
    return tok("<|im_start|>user\n", add_special_tokens=False).input_ids
def _q_ids(tok, q):
    return tok("\n\nQuestion: " + q + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
def has_alias(tok, ids, answers):
    s = tok.decode(ids).lower(); return any(a and a.lower() in s for a in answers)

def snap(CAP):
    # mean over layers of per-layer mean
    n = len(CAP)
    if not n: return None
    s0 = sum(c[0]/c[3] for c in CAP.values())/n
    s1 = sum(c[1]/c[3] for c in CAP.values())/n
    p0 = sum(c[2]/c[3] for c in CAP.values())/n
    return s0, s1, p0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="2wikimqa")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--min_depth", type=int, default=1)
    ap.add_argument("--max_ctx", type=int, default=8000)
    ap.add_argument("--bench", default="/home/tiger/sprag-main/data/benchmarks/longbench_v1/data")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok, text_cfg = load_model()
    dev = model.get_input_embeddings().weight.device
    recs = [json.loads(l) for l in open(Path(args.bench) / f"{args.data}.jsonl")][: args.limit]
    agg = {"dup": [0.0, 0.0, 0.0, 0], "nodup": [0.0, 0.0, 0.0, 0]}  # m0M,mM2M,mp0,n
    gi = -1; nrec = 0
    for ri, rec in enumerate(recs):
        gi += 1
        if (gi % args.num_shards) != args.shard_id: continue
        ctx, q, answers = rec["context"], rec["input"], rec["answers"]
        doc_ids = _prefix_ids(tok) + tok(ctx, add_special_tokens=False).input_ids[: args.max_ctx]
        chunks = split_into_chunks(torch.tensor(doc_ids), chunk_size=args.chunk_size)
        t = -1
        for c in chunks:
            if c.chunk_id >= args.min_depth and has_alias(tok, doc_ids[c.a_start:c.a_end], answers):
                t = c.chunk_id; break
        if t < 1: continue  # need chunk0 in ctx at c100 (t>=1 so range(0,t) includes 0)
        ctx_idx = list(range(0, t))  # c100
        q_ids = _q_ids(tok, q)
        for kind in ("dup", "nodup"):
            if kind == "dup":
                ranges = [(0, args.M)] + [(chunks[i].a_start, chunks[i].a_end) for i in ctx_idx] + [(chunks[t].a_start, chunks[t].a_end)]
            else:
                ranges = [(chunks[i].a_start, chunks[i].a_end) for i in ctx_idx] + [(chunks[t].a_start, chunks[t].a_end)]
            asm = []
            for (a0, a1) in ranges: asm += doc_ids[a0:a1]
            seq = asm + q_ids
            CAP.clear()
            orig = patch(model, args.M)
            with torch.no_grad():
                model(input_ids=torch.tensor([seq], device=dev), use_cache=False)
            unpatch(model, orig)
            s = snap(CAP)
            if s:
                agg[kind][0] += s[0]; agg[kind][1] += s[1]; agg[kind][2] += s[2]; agg[kind][3] += 1
        nrec += 1
        if nrec % 5 == 0:
            print(f"[{nrec}] dup n={agg['dup'][3]} nodup n={agg['nodup'][3]}", flush=True)
    out = {"data": args.data, "M": args.M, "n": agg["dup"][3]}
    for kind in ("dup", "nodup"):
        n = max(agg[kind][3], 1)
        out[kind] = {"mass_0_M": agg[kind][0]/n, "mass_M_2M": agg[kind][1]/n, "mass_pos0": agg[kind][2]/n}
    out["dup"]["sink_total(0_2M)"] = out["dup"]["mass_0_M"] + out["dup"]["mass_M_2M"]
    out["nodup"]["sink_total(0_M)"] = out["nodup"]["mass_0_M"]
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("RESULT", json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
