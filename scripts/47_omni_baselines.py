#!/usr/bin/env python3
"""Faithful ReKV / MuKV baselines vs position-preserving full-KV reuse (ours).

All arms start from the SAME full prebaked KV (one forward over the whole clip +
question, hooks capture pre-RoPE post-norm Q/K/V; GATE1 proved we can rebuild the
exact post-RoPE cache at ANY positions via the model's own rotary). They differ on
two axes only -- which video t-groups are kept (SELECTION) and at which positions
(POSITION) -- isolating ReKV's two signatures (per-layer query retrieval + InfLLM
repositioning) against ours (content-agnostic keep at original gapped positions).

Arms scored per (sample, coverage) by gold-answer NLL (MC-by-PPL):
  fresh         : re-encode the kept subset (reference; recompute, not reuse)
  ours          : uniform/center select  + ORIGINAL gapped positions
  ours_compact  : uniform/center select  + COMPACT positions (gaps removed)
  rekv_origpos  : per-layer query-retrieval select + ORIGINAL positions
  rekv          : per-layer query-retrieval select + COMPACT positions (full ReKV)
  mukv          : multi-grained dual-signal (attn+fft) token compression to budget

cov100 identity: keep all groups => compact==original==full-clip forward (gate).

Env: DATASET, EGO_VIDEO_DIR, EGO_META, OMNI_DIR, REKV_SINK_G, REKV_LOCAL_FRAC,
     ARMS (comma list; default all), MUKV_ALPHA.
"""
import os, sys, json, argparse, importlib.util, tempfile, traceback
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        os.environ.pop(_k, None)
import numpy as np
import torch

SCRIPTS = "/home/tiger/sprag-main/scripts"
sys.path.insert(0, SCRIPTS)
spec = importlib.util.spec_from_file_location("cov44", os.path.join(SCRIPTS, "44_omni_coverage.py"))
cov = importlib.util.module_from_spec(spec); spec.loader.exec_module(cov)
omni_kv = cov.omni_kv
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M

REKV_SINK_G = int(os.environ.get("REKV_SINK_G", "1"))        # sink t-groups
REKV_LOCAL_FRAC = float(os.environ.get("REKV_LOCAL_FRAC", "0.25"))  # local window frac of budget
MUKV_ALPHA = float(os.environ.get("MUKV_ALPHA", "0.7"))     # attn vs fft weight
ARMS = os.environ.get("ARMS", "fresh,ours,ours_compact,rekv_origpos,rekv,mukv").split(",")
COVERAGE_MODE = os.environ.get("COVERAGE_MODE", "uniform")


# ---------------------------------------------------------------------------
def find_text_attn_and_rotary(thinker):
    attn = [m for m in thinker.modules() if hasattr(m, "q_norm") and hasattr(m, "k_norm")]
    rot = [m for m in thinker.modules() if hasattr(m, "apply_interleaved_mrope")]
    return attn, rot[0]


def capture(thinker, attn_mods, fwd, pos):
    """One forward: hooks grab pre-RoPE post-norm q/k/v per layer; also returns the
    model's own post-RoPE cache (use_cache=True) for the ours/fresh arms."""
    caps = [{} for _ in attn_mods]; hooks = []
    def mk(i, nm):
        def h(mod, inp, out): caps[i][nm] = out.detach()
        return h
    for i, a in enumerate(attn_mods):
        hooks.append(a.q_norm.register_forward_hook(mk(i, "q")))
        hooks.append(a.k_norm.register_forward_hook(mk(i, "k")))
        hooks.append(a.v_proj.register_forward_hook(mk(i, "v")))
    with torch.no_grad():
        out = thinker(**fwd, position_ids=pos, use_cache=True, return_dict=True)
    for h in hooks: h.remove()
    return caps, out.past_key_values


def rotate_k_layer(rotary, k_bshd, pos3d, dtype, dev):
    """k_bshd:[B,S,Hkv,D] -> post-RoPE [B,Hkv,S,D] at positions pos3d:[3,B,S]."""
    x = torch.zeros(1, 1, 1, dtype=dtype, device=dev)
    cos, sin = rotary(x, pos3d)
    k = k_bshd.transpose(1, 2)
    _, k_rot = M.apply_rotary_pos_emb(k[:, :1], k, cos, sin)
    return k_rot


def v_layer(v_flat, head_dim):
    """v_proj out [B,S,Hkv*D] -> [B,Hkv,S,D]."""
    B, S, _ = v_flat.shape
    return v_flat.view(B, S, -1, head_dim).transpose(1, 2)


# ---------------------------------------------------------------------------
def compact_positions(pos, span, ranges, kept_groups, t_grid):
    """Return a NEW [3,1,T] position tensor with dropped t-groups' gaps removed
    (InfLLM compaction). kept video groups renumbered to contiguous t; trailing
    text shifted down by total dropped groups. Tokens of dropped groups keep junk
    positions (they are never selected into a cache)."""
    new = pos.clone()
    kept = set(kept_groups)
    dropped_before = [sum(1 for g in range(gi) if g not in kept) for gi in range(t_grid)]
    total_dropped = sum(1 for g in range(t_grid) if g not in kept)
    lo, hi = span
    for gi in range(t_grid):
        a, b = ranges[gi]
        new[0, 0, a:b] = pos[0, 0, a:b] - dropped_before[gi]   # shift t row only
    # trailing tokens after the video block: text => t=h=w, shift all 3 rows
    for r in range(3):
        new[r, 0, hi:] = pos[r, 0, hi:] - total_dropped
    return new


def build_cache(caps, rotary, keep_idx, pos3d, head_dim, dtype, dev):
    """Per-layer (or shared) keep + positions -> post-RoPE DynamicCache.
    keep_idx: LongTensor (shared) OR list[LongTensor] per layer.
    pos3d:    [3,1,T] (shared) OR list per layer."""
    n = len(caps)
    per_layer_keep = isinstance(keep_idx, list)
    per_layer_pos = isinstance(pos3d, list)
    layers = []
    for i in range(n):
        ki = keep_idx[i] if per_layer_keep else keep_idx
        pi = pos3d[i] if per_layer_pos else pos3d
        k_rot = rotate_k_layer(rotary, caps[i]["k"], pi, dtype, dev)   # [1,Hkv,T,D]
        v = v_layer(caps[i]["v"], head_dim)                            # [1,Hkv,T,D]
        idx = ki.to(k_rot.device)
        layers.append((k_rot.index_select(2, idx).contiguous(),
                       v.index_select(2, idx).contiguous()))
    return omni_kv.build_cache_from_layers(layers)


# ---------------------------------------------------------------------------
def rekv_select_layer(caps_i, ranges, kept_budget_groups, q_idx, t_grid, head_dim):
    """ReKV per-layer retrieval: sink + local window + top-k query-relevant groups,
    total == kept_budget_groups. Returns sorted list of group indices."""
    k = caps_i["k"][0]      # [S,Hkv,D]
    q = caps_i["q"][0]      # [S,Hq,D]
    Hkv = k.shape[1]; Hq = q.shape[1]; grp = Hq // Hkv
    # GQA-expand K to query heads, flatten heads*dim
    k_exp = k.repeat_interleave(grp, dim=1)               # [S,Hq,D]
    qrep = q[q_idx].mean(0).reshape(-1).float()           # [Hq*D]
    sims = torch.empty(t_grid)
    for g in range(t_grid):
        a, b = ranges[g]
        brep = k_exp[a:b].mean(0).reshape(-1).float()     # [Hq*D]
        sims[g] = torch.dot(qrep, brep)
    sink = list(range(min(REKV_SINK_G, t_grid)))
    n_local = max(1, int(round(REKV_LOCAL_FRAC * kept_budget_groups)))
    local = list(range(max(0, t_grid - n_local), t_grid))
    fixed = set(sink) | set(local)
    need = kept_budget_groups - len(fixed)
    if need > 0:
        cand = [g for g in range(t_grid) if g not in fixed]
        cand.sort(key=lambda g: sims[g].item(), reverse=True)
        fixed |= set(cand[:need])
    elif need < 0:                                        # budget < sink+local: keep top sims among fixed
        keep = sorted(fixed, key=lambda g: sims[g].item(), reverse=True)[:kept_budget_groups]
        fixed = set(keep)
    return sorted(fixed)


def mukv_compress_layer(caps_i, span, ranges, kept_budget_groups, q_idx, t_grid, head_dim):
    """MuKV dual-signal token compression to ~budget video tokens. Score each video
    token by alpha*attn_indicator + (1-alpha)*fft_indicator, keep top-rho. Returns
    a LongTensor of kept VIDEO token indices (within span)."""
    k = caps_i["k"][0]                       # [S,Hkv,D]
    q = caps_i["q"][0]                        # [S,Hq,D]
    Hkv = k.shape[1]; Hq = q.shape[1]; grp = Hq // Hkv
    lo, hi = span
    kv = k[lo:hi]                            # [V,Hkv,D]
    kexp = kv.repeat_interleave(grp, dim=1)  # [V,Hq,D]
    # attention indicator: question-query attention mass onto each video token
    qrep = q[q_idx].mean(0)                   # [Hq,D]
    attn = torch.einsum("vhd,hd->v", kexp.float(), qrep.float())  # [V]
    # frequency indicator: mean magnitude of FFT over key dim, averaged across heads
    fft = torch.fft.rfft(kv.float(), dim=-1).abs().mean(dim=(1, 2))  # [V]
    def mm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)
    score = MUKV_ALPHA * mm(attn) + (1 - MUKV_ALPHA) * mm(fft)
    budget_tokens = int(round((kept_budget_groups / t_grid) * (hi - lo)))
    budget_tokens = max(1, budget_tokens)
    top = torch.topk(score, min(budget_tokens, score.numel())).indices
    return (top + lo).sort().values


# ---------------------------------------------------------------------------
def run_sample(eng, attn_mods, rotary, frames, question, options, gold, coverages):
    inputs = eng.build(frames, question)
    fwd = eng.fwd_kwargs(inputs); pos = eng.rope(inputs)
    caps, full_cache = capture(eng.thinker, attn_mods, fwd, pos)
    head_dim = caps[0]["k"].shape[-1]
    idrow = inputs["input_ids"][0].tolist(); T = len(idrow)
    span = omni_kv.video_token_span(idrow, eng.VID)
    t_grid = int(inputs["video_grid_thw"][0][0].item())
    ranges = omni_kv.tgroup_ranges(span, t_grid)
    q_idx = list(range(span[1], T))           # trailing text (question + stub) = retrieval query
    prefix = "\n"; score_opts = [" " + o for o in options]
    res = {"t_grid": t_grid, "T": T, "rows": []}

    def score(cache, kmax):
        nll = omni_kv.mc_option_nll(eng.thinker, eng.tok, cache, kmax, prefix, score_opts, eng.dev)
        return nll, int(np.argmin(nll))

    for cov in coverages:
        c = cov / 100.0
        groups = omni_kv.select_coverage_groups(t_grid, c, mode=COVERAGE_MODE)
        budget = len(groups)
        keep_uniform = omni_kv.build_keep_idx(T, span, ranges, groups, eng.dev)
        kmax_orig = int(pos[:, 0, :][:, keep_uniform].max().item())
        row = {"cov": cov, "n_groups": budget}

        # ---- ours: uniform select, original positions (== gather_kv on full cache)
        if "ours" in ARMS:
            cache = omni_kv.gather_kv(full_cache, keep_uniform, device=None)
            nll, pred = score(cache, kmax_orig)
            row["ours"] = {"nll": nll[gold], "acc": int(pred == gold)}

        # ---- ours_compact: uniform select, compacted positions
        pos_c = compact_positions(pos, span, ranges, groups, t_grid)
        kmax_c = int(pos_c[:, 0, :][:, keep_uniform].max().item())
        if "ours_compact" in ARMS:
            cache = build_cache(caps, rotary, keep_uniform, pos_c, head_dim, eng.mdtype, eng.dev)
            nll, pred = score(cache, kmax_c)
            row["ours_compact"] = {"nll": nll[gold], "acc": int(pred == gold)}

        # ---- ReKV: per-layer retrieval select
        if "rekv" in ARMS or "rekv_origpos" in ARMS:
            sel = [rekv_select_layer(caps[i], ranges, budget, q_idx, t_grid, head_dim)
                   for i in range(len(caps))]
            keep_layers_orig = [omni_kv.build_keep_idx(T, span, ranges, sel[i], eng.dev)
                                for i in range(len(caps))]
            if "rekv_origpos" in ARMS:
                cache = build_cache(caps, rotary, keep_layers_orig, pos, head_dim, eng.mdtype, eng.dev)
                # max original pos across union of kept (text always kept => same trailing max)
                nll, pred = score(cache, kmax_orig)
                row["rekv_origpos"] = {"nll": nll[gold], "acc": int(pred == gold)}
            if "rekv" in ARMS:
                pos_layers = [compact_positions(pos, span, ranges, sel[i], t_grid) for i in range(len(caps))]
                keep_layers = [omni_kv.build_keep_idx(T, span, ranges, sel[i], eng.dev) for i in range(len(caps))]
                cache = build_cache(caps, rotary, keep_layers, pos_layers, head_dim, eng.mdtype, eng.dev)
                # compact trailing max identical across layers (same #groups dropped count? NO: per-layer)
                # text trailing shift = total dropped; equal #kept groups => equal total dropped => same kmax
                kmax_rk = int(pos_layers[0][:, 0, :][:, keep_layers[0]].max().item())
                nll, pred = score(cache, kmax_rk)
                row["rekv"] = {"nll": nll[gold], "acc": int(pred == gold)}

        # ---- MuKV: dual-signal token compression (per layer)
        if "mukv" in ARMS:
            base_keep = set(range(T)) - set(range(span[0], span[1]))   # all non-video tokens
            keep_layers = []
            for i in range(len(caps)):
                vk = mukv_compress_layer(caps[i], span, ranges, budget, q_idx, t_grid, head_dim)
                ks = sorted(base_keep | set(vk.tolist()))
                keep_layers.append(torch.tensor(ks, dtype=torch.long, device=eng.dev))
            cache = build_cache(caps, rotary, keep_layers, pos, head_dim, eng.mdtype, eng.dev)
            nll, pred = score(cache, kmax_orig)
            row["mukv"] = {"nll": nll[gold], "acc": int(pred == gold)}

        # ---- fresh reference (recompute subset at gapped positions)
        if "fresh" in ARMS:
            kept_frames = []
            for g in groups:
                kept_frames += [frames[2 * g], frames[2 * g + 1]]
            finp = eng.build(kept_frames, question)
            fids = finp["input_ids"][0].tolist()
            cached_ids = [idrow[i] for i in keep_uniform.tolist()]
            fpos = pos[:, :, keep_uniform] if fids == cached_ids else eng.rope(finp)
            ffwd = eng.fwd_kwargs(finp)
            with torch.no_grad():
                fcache = eng.thinker(**ffwd, position_ids=fpos, use_cache=True, return_dict=True).past_key_values
            fmax = int(fpos[:, 0, :].max().item())
            nll, pred = score(fcache, fmax)
            row["fresh"] = {"nll": nll[gold], "acc": int(pred == gold), "pos_matched": bool(fids == cached_ids)}

        res["rows"].append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--coverages", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--out", default="/home/tiger/data/omni_baselines.json")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    if args.n_frames % 2: args.n_frames += 1
    out = args.out if args.num_shards == 1 else f"{args.out}.s{args.shard_id}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {r["uid"]: r for r in json.load(open(out))} if os.path.exists(out) else {}
    if done: print(f"resume: {len(done)} done in {out}")

    df = cov.load_meta(); vmap = cov.build_video_map(cov.VIDEO_DIR); avail = set(vmap)
    print(f"dataset={cov.DATASET} videos={len(avail)} arms={ARMS} mode={COVERAGE_MODE}", flush=True)
    eng = cov.Engine(); print("engine ready", eng.dev, flush=True)
    attn_mods, rotary = find_text_attn_and_rotary(eng.thinker)
    print(f"text-attn layers={len(attn_mods)}", flush=True)

    results = list(done.values()); n_new = 0
    for gi, (_, r0) in enumerate(df.iterrows()):
        if gi % args.num_shards != args.shard_id: continue
        uid, vidkey, question, options, gold = cov.parse_row(r0)
        if uid in done or n_new >= args.limit or vidkey not in avail: continue
        try:
            frames = cov.extract_frames(vmap[vidkey], args.n_frames, cov.FRAME_DIR)
            r = run_sample(eng, attn_mods, rotary, frames, question, options, gold, args.coverages)
            r["uid"] = uid; r["gold"] = gold
            results.append(r); n_new += 1
            cov100 = next((x for x in r["rows"] if x["cov"] == 100), {})
            print(f"[{n_new}] {uid} t_grid={r['t_grid']} cov100={{k:v['nll'] for k,v in cov100.items() if isinstance(v,dict)}}", flush=True)
            tmp = out + ".tmp"; json.dump(results, open(tmp, "w")); os.replace(tmp, out)
        except Exception:
            print(f"FAIL {uid}:"); traceback.print_exc()

    # summary
    print(f"\n==== SUMMARY n={n_new} ====")
    arms = [a for a in ARMS]
    for cov_ in args.coverages:
        line = f"cov{cov_:3d}: "
        for a in arms:
            nll = [rw[a]["nll"] for r in results for rw in r["rows"] if rw["cov"] == cov_ and a in rw]
            acc = [rw[a]["acc"] for r in results for rw in r["rows"] if rw["cov"] == cov_ and a in rw]
            if nll: line += f"{a}={np.mean(nll):.3f}/{np.mean(acc):.2f} "
        print(line, flush=True)
    print("RUNNER_DONE", flush=True)


if __name__ == "__main__":
    main()
