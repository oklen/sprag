#!/usr/bin/env python3
"""COV100 alpha=0 identity gate for the Qwen3-Omni video KV-splice engine.

The single most important check before any coverage curve: splicing the FULL
prebaked cache at 100% coverage must reproduce a plain full forward's logits.
Here we verify the minimal form -- feed only the last token against the cache of
the first T-1 (all spliced verbatim from the full multimodal forward) and require
the last logit row to match the reference to < tol. If this fails, the M-RoPE /
DynamicCache surgery is wrong.

Run ON a worker GPU after weights staged:
  CUDA_VISIBLE_DEVICES=0 OMNI_DIR=/tmp/Qwen3-Omni-30B python 01_sanity_gate.py
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
        arr = np.full((h, w, 3), val, dtype=np.uint8)
        arr[:, :, 1] = 255 - val
        p = os.path.join(WORK, f"frame_{i:03d}.png")
        Image.fromarray(arr).save(p)
        frames.append("file://" + p)
    t = np.arange(sec * sr) / sr
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wp = os.path.join(WORK, "tone.wav")
    sf.write(wp, wav, sr)
    return frames, wp


def main():
    frames, wav = make_tiny_inputs()
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "audio", "audio": wav},
        {"type": "text", "text": "What happens in the clip?"},
    ]}]

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

    # build forward kwargs (accepted keys, floats -> bf16)
    accepted = set(inspect.signature(thinker.forward).parameters.keys())
    fwd = {}
    for k, v in inputs.items():
        if k in accepted and torch.is_tensor(v):
            v = v.to(dev)
            fwd[k] = v.to(mdtype) if v.is_floating_point() else v
    fwd["use_audio_in_video"] = USE_AV

    # 3D M-RoPE positions
    gri = thinker.get_rope_index
    pos, delta = gri(
        input_ids=inputs["input_ids"].to(dev),
        video_grid_thw=inputs.get("video_grid_thw").to(dev) if "video_grid_thw" in inputs else None,
        attention_mask=inputs["attention_mask"].to(dev),
        use_audio_in_video=USE_AV,
        audio_seqlens=inputs["feature_attention_mask"].sum(-1).to(dev) if "feature_attention_mask" in inputs else None,
        second_per_grids=inputs.get("video_second_per_grid").to(dev) if "video_second_per_grid" in inputs else None,
    )
    print("position_ids:", tuple(pos.shape), "T:", inputs["input_ids"].shape[-1])

    diff, ok = omni_kv.identity_gate(thinker, fwd, pos, dev, tol=2e-2)
    print(f"\n[IDENTITY GATE] max_abs_logit_diff = {diff:.6f}  tol=2e-2  -> {'PASS' if ok else 'FAIL'}")
    print("GATE_RESULT=", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
