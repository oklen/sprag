#!/usr/bin/env python3
"""Layout probe v2: verify the frame->token mapping on a REAL N-frame video.

The coverage runner selects video frames at TEMPORAL-GROUP granularity (Qwen's
temporal_patch_size=2 merges 2 frames -> 1 t-group). This probe processes a real
EgoSchema clip at N frames and dumps, so the runner's mapping is exact:
  * video_grid_thw = [T_grid, H, W]; T_grid = #t-groups
  * the contiguous [start,end) span of video tokens in input_ids
  * tokens-per-t-group = span_len / T_grid  (must divide evenly)
  * the t-row of position_ids across the span (each t-group => constant t)
  * the resulting map: t-group g -> token range [start+g*tpg, start+(g+1)*tpg)

Run ON a worker GPU after a video is available:
  CUDA_VISIBLE_DEVICES=0 OMNI_DIR=/tmp/Qwen3-Omni-30B \
    EGO_VIDEO=/tmp/egoschema_videos/<uid>.mp4 N_FRAMES=64 python 45_omni_layout_probe.py
"""
import os, glob, inspect
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        os.environ.pop(_k, None)
import torch

MODEL_DIR = os.environ.get("OMNI_DIR", "/tmp/Qwen3-Omni-30B")
N_FRAMES = int(os.environ.get("N_FRAMES", "64"))
USE_AV = os.environ.get("USE_AV", "0") == "1"


def find_video():
    v = os.environ.get("EGO_VIDEO")
    if v and os.path.exists(v):
        return v
    cands = sorted(glob.glob("/tmp/egoschema_videos/**/*.mp4", recursive=True))
    if not cands:
        raise SystemExit("no video found under /tmp/egoschema_videos")
    return cands[0]


def main():
    vid = find_video()
    print("video:", vid, "| N_FRAMES:", N_FRAMES, "| USE_AV:", USE_AV)
    content = [{"type": "video", "video": vid, "nframes": N_FRAMES}]
    if USE_AV:
        content[0]["video"] = vid  # audio muxed handled by use_audio_in_video flag
    content.append({"type": "text", "text": "Answer the question."})
    messages = [{"role": "user", "content": content}]

    from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeForConditionalGeneration
    from qwen_omni_utils import process_mm_info
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL_DIR)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AV)
    print("videos:", None if videos is None else (len(videos), tuple(videos[0].shape)))
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=USE_AV)
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    grid = inputs["video_grid_thw"][0].tolist()
    print("video_grid_thw:", grid, "(T_grid, H, W)")
    T_grid = grid[0]

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    thinker = model.thinker.eval()
    dev = next(thinker.parameters()).device
    cfg = thinker.config
    def cid(name):
        return getattr(cfg, name, None) or getattr(getattr(cfg, "thinker_config", cfg), name, None)
    VID = cid("video_token_id"); AUD = cid("audio_token_id")
    VS, VE = cid("vision_start_token_id"), cid("vision_end_token_id")

    ids = inputs["input_ids"][0].tolist()
    vpos = [i for i, x in enumerate(ids) if x == VID]
    apos = [i for i, x in enumerate(ids) if x == AUD]
    span = (vpos[0], vpos[-1] + 1) if vpos else None
    print("\nseq len:", len(ids))
    print("video token span:", span, "n_video_tok:", len(vpos),
          "contiguous:", (span and (span[1] - span[0]) == len(vpos)))
    print("vision_start at:", [i for i, x in enumerate(ids) if x == VS],
          "vision_end at:", [i for i, x in enumerate(ids) if x == VE])
    print("audio tokens:", len(apos), "span:", (apos[0], apos[-1] + 1) if apos else None)

    if span:
        span_len = span[1] - span[0]
        tpg = span_len / T_grid
        print(f"tokens-per-t-group = {span_len}/{T_grid} = {tpg}  (integer: {span_len % T_grid == 0})")

    # positions
    gri = thinker.get_rope_index
    pos, delta = gri(
        input_ids=inputs["input_ids"].to(dev),
        video_grid_thw=inputs["video_grid_thw"].to(dev),
        attention_mask=inputs["attention_mask"].to(dev),
        use_audio_in_video=USE_AV,
        audio_seqlens=inputs["feature_attention_mask"].sum(-1).to(dev) if "feature_attention_mask" in inputs else None,
        second_per_grids=inputs["video_second_per_grid"].to(dev) if "video_second_per_grid" in inputs else None)
    print("position_ids:", tuple(pos.shape), "mrope_delta:", delta.flatten().tolist()[:1])
    if span:
        trow = pos[0, 0, span[0]:span[1]].tolist()
        # distinct t-values across the video span and run-lengths
        runs = []
        cur, n = trow[0], 0
        for t in trow:
            if t == cur:
                n += 1
            else:
                runs.append((cur, n)); cur, n = t, 1
        runs.append((cur, n))
        print("distinct t-values in span:", len(runs), "(expect == T_grid =", T_grid, ")")
        print("first 5 t-runs (t,count):", runs[:5])
        print("t-run counts all-equal:", len(set(r[1] for r in runs)) == 1)
        print("EXAMPLE map t-group->token-range (first 3):")
        tpg_i = (span[1] - span[0]) // T_grid
        for g in range(min(3, T_grid)):
            print(f"  group {g}: tokens [{span[0]+g*tpg_i}, {span[0]+(g+1)*tpg_i})  t={runs[g][0] if g<len(runs) else '?'}")
    print("\nLAYOUT_PROBE_DONE")


if __name__ == "__main__":
    main()
