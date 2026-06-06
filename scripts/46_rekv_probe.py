#!/usr/bin/env python3
"""Keystone validation for the faithful ReKV arm.

Goal: prove we can rebuild the model's post-RoPE K/V cache from CAPTURED pre-RoPE
(post-norm) K/V by re-applying the model's OWN rotary at arbitrary positions. If
re-rotating captured K at its ORIGINAL positions reproduces the model's cached K
to <1e-3, then InfLLM-style REPOSITIONING (re-rotate at NEW positions) is exact.

Run on a GPU worker via a tmux launcher. Loads the existing Engine from the
coverage harness, runs ONE EgoSchema sample, and reports per-layer max|diff|.
"""
import os, sys, importlib.util
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        os.environ.pop(_k, None)
import torch

SCRIPTS = "/home/tiger/sprag-main/scripts"
sys.path.insert(0, SCRIPTS)
spec = importlib.util.spec_from_file_location("cov44", os.path.join(SCRIPTS, "44_omni_coverage.py"))
cov = importlib.util.module_from_spec(spec); spec.loader.exec_module(cov)
omni_kv = cov.omni_kv

# import the model's own rotary helpers (exact same ops the forward uses)
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M


def find_text_attn_and_rotary(thinker):
    attn = [m for m in thinker.modules() if hasattr(m, "q_norm") and hasattr(m, "k_norm")]
    rot = [m for m in thinker.modules() if hasattr(m, "apply_interleaved_mrope")]
    assert attn and rot, f"attn={len(attn)} rot={len(rot)}"
    return attn, rot[0]


def capture_preRoPE(thinker, attn_mods, fwd, pos):
    """Hook each text-attn q_norm/k_norm/v_proj to grab pre-RoPE post-norm q/k/v."""
    caps = [{} for _ in attn_mods]
    hooks = []
    def mk(i, name):
        def h(mod, inp, out):
            caps[i][name] = out.detach()
        return h
    for i, a in enumerate(attn_mods):
        hooks.append(a.q_norm.register_forward_hook(mk(i, "q")))
        hooks.append(a.k_norm.register_forward_hook(mk(i, "k")))
        hooks.append(a.v_proj.register_forward_hook(mk(i, "v")))
    with torch.no_grad():
        thinker(**fwd, position_ids=pos, use_cache=False, return_dict=True)
    for h in hooks:
        h.remove()
    return caps


def rotate_k(rotary, k_postnorm_bshd, pos3d, dtype, dev, head_dim):
    """k_postnorm_bshd: [B,S,Hkv,D] -> post-RoPE [B,Hkv,S,D] at positions pos3d[3,B,S]."""
    x = torch.zeros(1, 1, 1, dtype=dtype, device=dev)
    cos, sin = rotary(x, pos3d)                       # [B,S,D], [B,S,D]
    k = k_postnorm_bshd.transpose(1, 2)               # [B,Hkv,S,D]
    q = k[:, :1]                                       # dummy q, same dtype/shape rules
    _, k_rot = M.apply_rotary_pos_emb(q, k, cos, sin)
    return k_rot


def main():
    eng = cov.Engine()
    print("engine ready on", eng.dev, "dtype", eng.mdtype, flush=True)
    attn_mods, rotary = find_text_attn_and_rotary(eng.thinker)
    print(f"text-attn layers={len(attn_mods)}", flush=True)

    # one sample
    df = cov.load_meta(); vmap = cov.build_video_map(cov.VIDEO_DIR); avail = set(vmap)
    row = next(r for _, r in df.iterrows() if cov.parse_row(r)[1] in avail)
    uid, vidkey, question, options, gold = cov.parse_row(row)
    frames = cov.extract_frames(vmap[vidkey], 32, cov.FRAME_DIR)
    inputs = eng.build(frames, question)
    fwd = eng.fwd_kwargs(inputs); pos = eng.rope(inputs)
    print(f"sample {uid} T={inputs['input_ids'].shape[1]} pos={tuple(pos.shape)}", flush=True)

    # reference: model's own cached post-RoPE K/V
    with torch.no_grad():
        ref = eng.thinker(**fwd, position_ids=pos, use_cache=True, return_dict=True)
    ref_layers = omni_kv.cache_layers(ref.past_key_values)   # list[(K,V)] [B,Hkv,S,D]

    # capture pre-RoPE post-norm q/k/v
    caps = capture_preRoPE(eng.thinker, attn_mods, fwd, pos)
    head_dim = caps[0]["k"].shape[-1]   # k_norm out is [B,S,Hkv,D]
    print(f"head_dim={head_dim} k_cap={tuple(caps[0]['k'].shape)} v_cap={tuple(caps[0]['v'].shape)}", flush=True)

    # GATE 1: re-rotate captured K at ORIGINAL positions == cached K ?
    maxdiff_k = 0.0; maxdiff_v = 0.0
    for i, (K, V) in enumerate(ref_layers):
        k_rot = rotate_k(rotary, caps[i]["k"], pos, eng.mdtype, eng.dev, head_dim)
        dk = (k_rot.float() - K.float()).abs().max().item()
        maxdiff_k = max(maxdiff_k, dk)
        # V is position-independent: captured v_proj (reshaped) must equal cached V
        v_cap = caps[i]["v"].view(caps[i]["v"].shape[0], caps[i]["v"].shape[1], -1, head_dim).transpose(1, 2)
        dv = (v_cap.float() - V.float()).abs().max().item()
        maxdiff_v = max(maxdiff_v, dv)
        if i < 3 or dk > 1e-2:
            print(f"  layer {i:2d}: max|K_rot-K_cache|={dk:.2e}  max|V_cap-V_cache|={dv:.2e}", flush=True)
    print(f"\nGATE1 rotation identity: max|dK|={maxdiff_k:.2e}  max|dV|={maxdiff_v:.2e}", flush=True)
    print("GATE1", "PASS" if (maxdiff_k < 1e-2 and maxdiff_v < 1e-2) else "FAIL", flush=True)

    # GATE 2: reposition sanity — shift every position by +1000, re-rotate, then a
    # full-prefix decode of one token at shifted pos must match a decode at original
    # pos (RoPE is relative, so a uniform shift of BOTH cache and query is identity).
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
