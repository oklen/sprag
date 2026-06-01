#!/usr/bin/env python3
"""Probe the Qwen3-Omni Thinker API for the video KV-cache splice engine.

Goal: before building the prebake/splice/sanity-gate engine, empirically nail
down (on a real GPU, with real weights) the exact API:
  * how the processor lays out video / audio / text tokens in input_ids,
  * the structure of the returned past_key_values (n_layers, K/V shapes, GQA),
  * how the 3D interleaved M-RoPE position_ids are produced (get_rope_index),
  * that a single Thinker forward with use_cache=True runs and is reproducible.

Uses a tiny synthetic input (a few PNG frames as a "video" + a short sine WAV)
so it needs no dataset. Video is passed as a FRAME LIST (file://...) to avoid
any codec dependency. Run ON a worker GPU after weights are staged to /tmp.

  CUDA_VISIBLE_DEVICES=0 python 00_omni_api_probe.py
"""
import os, sys, json, inspect, traceback

# Iron rule 3: strip proxies before any internal/model call.
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        os.environ.pop(_k, None)

import numpy as np
import torch

MODEL_DIR = os.environ.get("OMNI_DIR", "/tmp/Qwen3-Omni-30B")
WORK = os.environ.get("OMNI_WORK", "/tmp/omni_probe")
os.makedirs(WORK, exist_ok=True)


def banner(s):
    print("\n" + "=" * 8 + " " + s + " " + "=" * 8, flush=True)


def make_tiny_inputs(n_frames=8, w=64, h=64, sec=2, sr=16000):
    """Return (frame_paths:list[str], wav_path:str). No video codec needed."""
    from PIL import Image
    import soundfile as sf
    frames = []
    for i in range(n_frames):
        val = (i * 28) % 256
        arr = np.full((h, w, 3), val, dtype=np.uint8)
        arr[:, :, 1] = (255 - val)  # vary a channel so frames differ
        p = os.path.join(WORK, f"frame_{i:03d}.png")
        Image.fromarray(arr).save(p)
        frames.append("file://" + p)
    t = np.arange(sec * sr) / sr
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wp = os.path.join(WORK, "tone.wav")
    sf.write(wp, wav, sr)
    return frames, wp


def describe_kv(pkv):
    """Print structure of past_key_values (DynamicCache or legacy tuple)."""
    banner("past_key_values structure")
    print("type:", type(pkv))
    layers = None
    try:
        if hasattr(pkv, "key_cache"):
            layers = list(zip(pkv.key_cache, pkv.value_cache))
        elif hasattr(pkv, "layers"):
            layers = [(l.keys, l.values) for l in pkv.layers]
        elif isinstance(pkv, (list, tuple)):
            layers = pkv
    except Exception as e:
        print("  (could not iterate:", e, ")")
    if layers:
        print("  n_layers:", len(layers))
        k0, v0 = layers[0]
        print("  layer0 K:", tuple(k0.shape), k0.dtype, "| V:", tuple(v0.shape), v0.dtype)
        print("  layer-last K:", tuple(layers[-1][0].shape))
    return layers


def main():
    banner("build tiny inputs")
    frames, wav = make_tiny_inputs()
    print("frames:", len(frames), "wav:", wav)

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "audio", "audio": wav},
            {"type": "text", "text": "What happens in the clip?"},
        ],
    }]

    banner("load processor")
    from transformers import Qwen3OmniMoeProcessor
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL_DIR)
    print("processor:", type(proc).__name__)

    from qwen_omni_utils import process_mm_info
    USE_AV = False  # audio passed as a separate stream, not muxed-in-video
    audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AV)
    print("process_mm_info -> audios:", None if audios is None else len(audios),
          "images:", None if images is None else len(images),
          "videos:", None if videos is None else (len(videos), getattr(videos[0], "shape", None)))

    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=USE_AV)
    banner("processor output keys/shapes")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v)} {v}")

    banner("load model (full -> disable_talker -> thinker)")
    from transformers import Qwen3OmniMoeForConditionalGeneration
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0",
    )
    if hasattr(model, "disable_talker"):
        try:
            model.disable_talker()
            print("talker disabled")
        except Exception as e:
            print("disable_talker failed:", e)
    thinker = model.thinker
    thinker.eval()
    print("thinker:", type(thinker).__name__)
    print("mem GB:", round(torch.cuda.memory_allocated() / 1e9, 1))

    banner("thinker.forward signature")
    print(list(inspect.signature(thinker.forward).parameters.keys()))
    banner("get_rope_index signature")
    gri = getattr(thinker, "get_rope_index", None) or getattr(thinker.model, "get_rope_index", None)
    if gri is not None:
        print("found on:", gri.__qualname__)
        print(list(inspect.signature(gri).parameters.keys()))
    else:
        print("NO get_rope_index found")

    dev = next(thinker.parameters()).device
    mdtype = next(thinker.parameters()).dtype
    accepted = set(inspect.signature(thinker.forward).parameters.keys())
    fwd = {}
    for k, v in inputs.items():
        if k in accepted and torch.is_tensor(v):
            v = v.to(dev)
            if v.is_floating_point():
                v = v.to(mdtype)   # processor emits fp32; thinker is bf16
            fwd[k] = v
        elif k in accepted:
            fwd[k] = v
    fwd["use_audio_in_video"] = USE_AV
    print("\nforwarding with keys:", list(fwd.keys()), "| model dtype:", mdtype)

    banner("thinker forward (use_cache=True)")
    with torch.no_grad():
        out = thinker(**fwd, use_cache=True, return_dict=True)
    logits = out.logits
    print("logits:", tuple(logits.shape), logits.dtype)
    pkv = out.past_key_values
    describe_kv(pkv)

    banner("3D M-RoPE position_ids")
    if gri is not None:
        try:
            params = inspect.signature(gri).parameters
            # derived args get_rope_index needs that aren't raw processor keys
            derived = {
                "use_audio_in_video": USE_AV,
                "second_per_grids": inputs.get("video_second_per_grid"),
                "audio_seqlens": (inputs["feature_attention_mask"].sum(-1)
                                  if "feature_attention_mask" in inputs else None),
            }
            kw = {}
            for name in params:
                if name == "self":
                    continue
                if name in inputs:
                    v = inputs[name]
                    kw[name] = v.to(dev) if torch.is_tensor(v) else v
                elif name in derived and derived[name] is not None:
                    v = derived[name]
                    kw[name] = v.to(dev) if torch.is_tensor(v) else v
            print("calling get_rope_index with:", list(kw.keys()))
            res = gri(**kw)
            if isinstance(res, tuple):
                pos, delta = res[0], res[1] if len(res) > 1 else None
            else:
                pos, delta = res, None
            print("position_ids:", tuple(pos.shape), pos.dtype, "(dims=", pos.shape[0], ")")
            print("  t[:,:12]:", pos[0, 0, :12].tolist())
            print("  h[:,:12]:", pos[1, 0, :12].tolist() if pos.shape[0] > 1 else "n/a")
            print("  w[:,:12]:", pos[2, 0, :12].tolist() if pos.shape[0] > 2 else "n/a")
            print("  last cols t/h/w:",
                  pos[0, 0, -4:].tolist(),
                  pos[1, 0, -4:].tolist() if pos.shape[0] > 1 else [],
                  pos[2, 0, -4:].tolist() if pos.shape[0] > 2 else [])
            if delta is not None:
                print("  mrope_delta:", delta.tolist() if torch.is_tensor(delta) else delta)
        except Exception:
            print("get_rope_index call FAILED:")
            traceback.print_exc()

    banner("token layout (special-token positions)")
    cfg = thinker.config
    ids = inputs["input_ids"][0].tolist()
    sd = {}
    for name in ["video_token_id", "audio_token_id", "image_token_id",
                 "vision_start_token_id", "vision_end_token_id",
                 "audio_start_token_id", "audio_end_token_id"]:
        tid = getattr(cfg, name, None)
        if tid is None:
            tid = getattr(getattr(cfg, "thinker_config", cfg), name, None)
        if tid is not None:
            cnt = sum(1 for x in ids if x == tid)
            sd[name] = (tid, cnt)
    print(json.dumps(sd, indent=2))
    print("total seq len:", len(ids))

    banner("DONE - API probe ok")


if __name__ == "__main__":
    main()
