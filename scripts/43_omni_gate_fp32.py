#!/usr/bin/env python3
"""fp32 identity gate -- rigorous confirmation the splice math is correct.

In bf16 the model's own prefill-vs-decode logits differ ~0.4-0.9 (deterministic
kernel/reduction-order effect), too loose for a tight identity check. fp32 removes
that: a correct position-preserving splice must reproduce the one-shot full
forward to < 1e-3. 30B fp32 ~120GB -> spread over 2 GPUs (device_map=auto).

  CUDA_VISIBLE_DEVICES=0,1 OMNI_DIR=/tmp/Qwen3-Omni-30B python 01c_gate_fp32.py
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
        MODEL_DIR, dtype=torch.float32, attn_implementation="sdpa", device_map="auto")
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    thinker = model.thinker.eval()
    in_dev = thinker.get_input_embeddings().weight.device
    print("input device:", in_dev)
    accepted = set(inspect.signature(thinker.forward).parameters.keys())

    fwd = {}
    for k, v in inputs.items():
        if k in accepted and torch.is_tensor(v):
            v = v.to(in_dev); fwd[k] = v.to(torch.float32) if v.is_floating_point() else v
    fwd["use_audio_in_video"] = USE_AV

    pos, _ = thinker.get_rope_index(
        input_ids=inputs["input_ids"].to(in_dev),
        video_grid_thw=inputs["video_grid_thw"].to(in_dev),
        attention_mask=inputs["attention_mask"].to(in_dev),
        use_audio_in_video=USE_AV,
        audio_seqlens=inputs["feature_attention_mask"].sum(-1).to(in_dev),
        second_per_grids=inputs["video_second_per_grid"].to(in_dev))
    T = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        ref = thinker(**fwd, position_ids=pos, use_cache=True, return_dict=True)
    L_full = ref.logits[0, -1].float()
    cache_full = ref.past_key_values

    # spliced decode (keep per-layer device, model is split across GPUs)
    cache_sp = omni_kv.gather_kv(cache_full, torch.arange(T - 1, device=in_dev), device=None)
    last_id = fwd["input_ids"][:, -1:]
    last_pos = pos[:, :, -1:]
    with torch.no_grad():
        L_sp = thinker(input_ids=last_id, position_ids=last_pos,
                       past_key_values=cache_sp, use_cache=False, return_dict=True).logits[0, -1].float()

    diff = (L_full - L_sp).abs().max().item()
    print(f"\nlogit scale (full abs max): {L_full.abs().max().item():.4f}")
    print(f"[FP32 IDENTITY GATE] d_full_vs_sp = {diff:.8f}  tol=1e-3  -> {'PASS' if diff < 1e-3 else 'FAIL'}")
    print("top1 full/sp:", int(L_full.argmax()), int(L_sp.argmax()))
    print("GATE_RESULT=", "PASS" if diff < 1e-3 else "FAIL")


if __name__ == "__main__":
    main()
