#!/usr/bin/env python3
"""Diagnostic for the cov100 identity gate: separate splice-bug from bf16 noise.

Three last-row logit vectors from ONE model load:
  L_full : one-shot full forward over all T tokens (prefill kernel for last tok)
  L_nat  : forward first T-1 tokens natively -> cache_nat; decode last token
  L_sp   : full cache -> gather_kv(keep=arange(T-1)) -> cache_sp; decode last token

Deltas:
  d_full_vs_nat = |L_full - L_nat|  -> inherent prefill-vs-decode bf16 kernel noise
  d_nat_vs_sp   = |L_nat  - L_sp |  -> gather_kv / DynamicCache surgery fidelity  (MUST be ~0)
  d_full_vs_sp  = |L_full - L_sp |  -> what the simple gate measured

If d_nat_vs_sp ~ 0 while d_full_vs_* is the larger number, the splice is CORRECT
and the gate should compare against the native-decode reference, not the one-shot
prefill. Also reports top-1 token agreement.
"""
import os, inspect
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        os.environ.pop(_k, None)
import numpy as np
import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from sprag import omni_kv

MODEL_DIR = os.environ.get("OMNI_DIR", "/tmp/Qwen3-Omni-30B")
WORK = os.environ.get("OMNI_WORK", "/tmp/omni_probe")
os.makedirs(WORK, exist_ok=True)
USE_AV = False


def make_tiny_inputs(n_frames=8, w=64, h=64, sec=2, sr=16000):
    from PIL import Image
    import soundfile as sf
    frames = []
    for i in range(n_frames):
        val = (i * 28) % 256
        arr = np.full((h, w, 3), val, dtype=np.uint8); arr[:, :, 1] = 255 - val
        p = os.path.join(WORK, f"frame_{i:03d}.png"); Image.fromarray(arr).save(p)
        frames.append("file://" + p)
    t = np.arange(sec * sr) / sr
    sf.write(os.path.join(WORK, "tone.wav"),
             (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    return frames, os.path.join(WORK, "tone.wav")


def main():
    frames, wav = make_tiny_inputs()
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "audio", "audio": wav},
        {"type": "text", "text": "What happens in the clip?"}]}]
    from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeForConditionalGeneration
    from qwen_omni_utils import process_mm_info
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL_DIR)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AV)
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=USE_AV)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    thinker = model.thinker.eval()
    dev = next(thinker.parameters()).device
    mdtype = next(thinker.parameters()).dtype
    accepted = set(inspect.signature(thinker.forward).parameters.keys())

    fwd = {}
    for k, v in inputs.items():
        if k in accepted and torch.is_tensor(v):
            v = v.to(dev); fwd[k] = v.to(mdtype) if v.is_floating_point() else v
    fwd["use_audio_in_video"] = USE_AV

    pos, _ = thinker.get_rope_index(
        input_ids=inputs["input_ids"].to(dev),
        video_grid_thw=inputs["video_grid_thw"].to(dev),
        attention_mask=inputs["attention_mask"].to(dev),
        use_audio_in_video=USE_AV,
        audio_seqlens=inputs["feature_attention_mask"].sum(-1).to(dev),
        second_per_grids=inputs["video_second_per_grid"].to(dev))
    T = inputs["input_ids"].shape[-1]
    print("T:", T, "pos:", tuple(pos.shape))

    # 1) one-shot full
    with torch.no_grad():
        ref = thinker(**fwd, position_ids=pos, use_cache=True, return_dict=True)
    L_full = ref.logits[0, -1].float()
    cache_full = ref.past_key_values

    # 2) native first T-1 -> decode last
    fwd_n = dict(fwd)
    fwd_n["input_ids"] = fwd["input_ids"][:, :-1]
    if "attention_mask" in fwd_n:
        fwd_n["attention_mask"] = fwd["attention_mask"][:, :-1]
    with torch.no_grad():
        nato = thinker(**fwd_n, position_ids=pos[:, :, :-1], use_cache=True, return_dict=True)
    cache_nat = nato.past_key_values
    last_id = fwd["input_ids"][:, -1:]
    last_pos = pos[:, :, -1:]
    with torch.no_grad():
        L_nat = thinker(input_ids=last_id, position_ids=last_pos,
                        past_key_values=cache_nat, use_cache=False, return_dict=True).logits[0, -1].float()

    # 3) spliced (gather_kv identity selection) -> decode last
    cache_sp = omni_kv.gather_kv(cache_full, torch.arange(T - 1, device=dev), device=dev)
    with torch.no_grad():
        L_sp = thinker(input_ids=last_id, position_ids=last_pos,
                       past_key_values=cache_sp, use_cache=False, return_dict=True).logits[0, -1].float()

    # repeatability controls: same input twice -> noise floor
    with torch.no_grad():
        L_full2 = thinker(**fwd, position_ids=pos, use_cache=True, return_dict=True).logits[0, -1].float()
        cache_sp2 = omni_kv.gather_kv(cache_full, torch.arange(T - 1, device=dev), device=dev)
        L_sp2 = thinker(input_ids=last_id, position_ids=last_pos,
                        past_key_values=cache_sp2, use_cache=False, return_dict=True).logits[0, -1].float()

    def d(a, b): return (a - b).abs().max().item()
    print(f"\n--- repeatability (bf16 noise floor) ---")
    print(f"d_full_vs_full2 = {d(L_full, L_full2):.6f}   (same full forward, twice)")
    print(f"d_sp_vs_sp2     = {d(L_sp, L_sp2):.6f}   (same spliced decode, twice)")
    print(f"--- comparisons ---")
    print(f"d_full_vs_nat = {d(L_full, L_nat):.6f}   (prefill-vs-decode bf16 kernel noise)")
    print(f"d_nat_vs_sp   = {d(L_nat, L_sp):.6f}   (gather_kv splice fidelity -- MUST be ~0)")
    print(f"d_full_vs_sp  = {d(L_full, L_sp):.6f}   (simple gate metric)")
    print("top1 full/nat/sp:", int(L_full.argmax()), int(L_nat.argmax()), int(L_sp.argmax()))
    print("logit scale (full abs max):", L_full.abs().max().item())
    print("GATE_DIAG_DONE")


if __name__ == "__main__":
    main()
