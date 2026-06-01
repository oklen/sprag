#!/usr/bin/env python3
"""Video KV-cache Delta(coverage) curve on EgoSchema with Qwen3-Omni.

For each EgoSchema-Subset question (public GT), sample N frames, PREBAKE the full
clip once (global causal + cross-modal cache), then at each coverage c compare:

  CACHED arm: reuse the prebaked KV for the kept frames (+ all non-video tokens)
    at their ORIGINAL gapped M-RoPE positions (dropped frames simply absent).
  FRESH arm:  re-encode the SAME kept frames from scratch (sees only the subset),
    POSITION-MATCHED by overriding position_ids to the same gapped positions
    (asserted: fresh_ids == prebake_ids[keep_idx]).

Both arms then score the 5 options by mean gold-NLL (MC-by-PPL, argmin = pred).
Metrics per (sample, coverage): acc_cached, acc_fresh, gold_nll_cached/fresh.
Delta(c) = acc_cached - acc_fresh ; ParPPL. cov100 is a built-in sanity (cached
≈ fresh ≈ full clip). Checkpointed per-sample (atomic tmp+rename), resumable.

  CUDA_VISIBLE_DEVICES=0 OMNI_DIR=/tmp/Qwen3-Omni-30B \
    python 44_omni_coverage.py --limit 100 --n_frames 32 \
    --coverages 20 40 60 80 100 --out /home/tiger/data/omni_cov.json
"""
import os, sys, json, argparse, glob, tempfile, traceback
from pathlib import Path
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        os.environ.pop(_k, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from sprag import omni_kv

MODEL_DIR = os.environ.get("OMNI_DIR", "/tmp/Qwen3-Omni-30B")
VIDEO_DIR = os.environ.get("EGO_VIDEO_DIR", "/tmp/egoschema_videos")
META_DIR = os.environ.get("EGO_META", "/tmp/egoschema_meta")
FRAME_DIR = os.environ.get("FRAME_DIR", "/tmp/ego_frames")
USE_AV = os.environ.get("USE_AV", "0") == "1"  # mux audio from video (cross-modal)


# ----------------------------------------------------------------------------
def extract_frames(path, n, out_dir, size=336):
    """Seek-decode n evenly-spaced frames -> PNGs (even n for temporal pairing)."""
    import av
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(f)
    c = av.open(path)
    s = c.streams.video[0]
    dur = None
    if s.duration and s.time_base:
        dur = float(s.duration * s.time_base)
    elif c.duration:
        dur = c.duration / 1e6
    paths = []
    if dur and dur > 0:
        for j in range(n):
            t = dur * (j + 0.5) / n
            c.seek(int(t / s.time_base), stream=s)
            frame = next(c.decode(s))
            img = frame.to_image().resize((size, size))
            p = os.path.join(out_dir, f"f{j:03d}.png"); img.save(p)
            paths.append("file://" + p)
    else:  # no duration: decode all, sub-sample
        frames = [fr for fr in c.decode(s)]
        idxs = [int(round(i * (len(frames) - 1) / (n - 1))) for i in range(n)]
        for j, ix in enumerate(idxs):
            img = frames[ix].to_image().resize((size, size))
            p = os.path.join(out_dir, f"f{j:03d}.png"); img.save(p)
            paths.append("file://" + p)
    c.close()
    return paths


def load_subset():
    import pandas as pd
    p = os.path.join(META_DIR, "Subset", "test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    return df


def find_video(uid):
    for ext in (".mp4", ".MP4", ".webm", ".mkv"):
        hits = glob.glob(os.path.join(VIDEO_DIR, "**", uid + ext), recursive=True)
        if hits:
            return hits[0]
    return None


def fmt_question(q, options):
    # Options are NOT listed in the prompt: we score each full option text as the
    # continuation (MC-by-PPL, argmin NLL). Listing them would let the model copy.
    return f"Question: {q}\nAnswer:"


# ----------------------------------------------------------------------------
class Engine:
    def __init__(self):
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeForConditionalGeneration
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL_DIR)
        m = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")
        if hasattr(m, "disable_talker"):
            m.disable_talker()
        self.thinker = m.thinker.eval()
        self.dev = next(self.thinker.parameters()).device
        self.mdtype = next(self.thinker.parameters()).dtype
        import inspect
        self.accepted = set(inspect.signature(self.thinker.forward).parameters.keys())
        cfg = self.thinker.config
        self.VID = getattr(cfg, "video_token_id", None) or getattr(getattr(cfg, "thinker_config", cfg), "video_token_id")
        self.tok = self.proc.tokenizer

    def build(self, frames, question):
        from qwen_omni_utils import process_mm_info
        content = [{"type": "video", "video": frames}, {"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AV)
        text = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=USE_AV)
        return inputs

    def fwd_kwargs(self, inputs):
        fwd = {}
        for k, v in inputs.items():
            if k in self.accepted and torch.is_tensor(v):
                v = v.to(self.dev)
                fwd[k] = v.to(self.mdtype) if v.is_floating_point() else v
        fwd["use_audio_in_video"] = USE_AV
        return fwd

    def rope(self, inputs):
        return self.thinker.get_rope_index(
            input_ids=inputs["input_ids"].to(self.dev),
            video_grid_thw=inputs["video_grid_thw"].to(self.dev),
            attention_mask=inputs["attention_mask"].to(self.dev),
            use_audio_in_video=USE_AV,
            audio_seqlens=inputs["feature_attention_mask"].sum(-1).to(self.dev) if "feature_attention_mask" in inputs else None,
            second_per_grids=inputs["video_second_per_grid"].to(self.dev) if "video_second_per_grid" in inputs else None)[0]

    def prefill(self, fwd, pos):
        with torch.no_grad():
            out = self.thinker(**fwd, position_ids=pos, use_cache=True, return_dict=True)
        return out.past_key_values


# ----------------------------------------------------------------------------
def run_sample(eng, frames, question, options, gold, coverages):
    # --- prebake full clip + question ---
    inputs = eng.build(frames, question)
    fwd = eng.fwd_kwargs(inputs)
    pos = eng.rope(inputs)                              # [3,1,T]
    full_cache = eng.prefill(fwd, pos)
    idrow = inputs["input_ids"][0].tolist()
    T = len(idrow)
    span = omni_kv.video_token_span(idrow, eng.VID)
    t_grid = int(inputs["video_grid_thw"][0][0].item())
    ranges = omni_kv.tgroup_ranges(span, t_grid)
    # 1-token anchor ("\n") provides the boundary logit so the FIRST option token
    # is scored too; shared across options so the comparison stays fair.
    prefix = "\n"
    score_opts = [" " + o for o in options]   # leading space for clean tokenization
    res = {"t_grid": t_grid, "T": T, "span": span, "rows": []}

    for cov in coverages:
        c = cov / 100.0
        groups = omni_kv.select_coverage_groups(t_grid, c, mode="uniform")
        keep = omni_kv.build_keep_idx(T, span, ranges, groups, eng.dev)
        keep_max_pos = int(pos[:, 0, :][:, keep].max().item())

        # CACHED arm: splice kept tokens' KV at original positions
        cached = omni_kv.gather_kv(full_cache, keep, device=None)
        nll_c = omni_kv.mc_option_nll(eng.thinker, eng.tok, cached, keep_max_pos,
                                      prefix, score_opts, eng.dev)
        pred_c = int(np.argmin(nll_c))

        # FRESH arm: re-encode kept frames, position-matched to gapped positions
        kept_frames = []
        for g in groups:
            kept_frames += [frames[2 * g], frames[2 * g + 1]]  # t-group = 2 frames
        finp = eng.build(kept_frames, question)
        fids = finp["input_ids"][0].tolist()
        cached_ids = [idrow[i] for i in keep.tolist()]
        if fids == cached_ids:
            fpos = pos[:, :, keep]                      # gapped, matched
        else:
            fpos = eng.rope(finp)                       # fallback: natural positions
        ffwd = eng.fwd_kwargs(finp)
        fresh_cache = eng.prefill(ffwd, fpos)
        fmax_pos = int(fpos[:, 0, :].max().item())
        nll_f = omni_kv.mc_option_nll(eng.thinker, eng.tok, fresh_cache, fmax_pos,
                                      prefix, score_opts, eng.dev)
        pred_f = int(np.argmin(nll_f))

        res["rows"].append({
            "cov": cov, "n_groups": len(groups),
            "acc_cached": int(pred_c == gold), "acc_fresh": int(pred_f == gold),
            "gold_nll_cached": nll_c[gold], "gold_nll_fresh": nll_f[gold],
            "pred_cached": pred_c, "pred_fresh": pred_f,
            "pos_matched": bool(fids == cached_ids),
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--coverages", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--out", default="/home/tiger/data/omni_cov.json")
    args = ap.parse_args()
    if args.n_frames % 2:
        args.n_frames += 1  # keep even (temporal pairing)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        done = {r["uid"]: r for r in json.load(open(args.out))}
        print(f"resume: {len(done)} samples already done")

    df = load_subset()
    eng = Engine()
    print("engine ready on", eng.dev)

    results = list(done.values())
    n_new = 0
    for _, row in df.iterrows():
        uid = row["video_idx"]
        if uid in done:
            continue
        if n_new >= args.limit:
            break
        vid = find_video(uid)
        if vid is None:
            continue
        options = list(row["option"])
        gold = int(row["answer"])
        question = fmt_question(row["question"], options)
        try:
            frames = extract_frames(vid, args.n_frames, FRAME_DIR)
            r = run_sample(eng, frames, question, options, gold, args.coverages)
            r["uid"] = uid; r["gold"] = gold
            results.append(r); n_new += 1
            last = {row_["cov"]: (row_["acc_cached"], row_["acc_fresh"]) for row_ in r["rows"]}
            print(f"[{n_new}] {uid} t_grid={r['t_grid']} acc(cov->c/f)={last}", flush=True)
            tmp = args.out + ".tmp"
            json.dump(results, open(tmp, "w"))
            os.replace(tmp, args.out)
        except Exception:
            print(f"FAIL {uid}:"); traceback.print_exc()

    # summary
    print("\n==== SUMMARY (n=%d) ====" % n_new)
    covs = args.coverages
    for cov in covs:
        ac = [rw["acc_cached"] for r in results for rw in r["rows"] if rw["cov"] == cov]
        af = [rw["acc_fresh"] for r in results for rw in r["rows"] if rw["cov"] == cov]
        gc = [rw["gold_nll_cached"] for r in results for rw in r["rows"] if rw["cov"] == cov]
        gf = [rw["gold_nll_fresh"] for r in results for rw in r["rows"] if rw["cov"] == cov]
        if ac:
            print(f"cov{cov:3d}: acc_cached={np.mean(ac):.3f} acc_fresh={np.mean(af):.3f} "
                  f"Dacc={np.mean(ac)-np.mean(af):+.3f} | nll_c={np.mean(gc):.3f} "
                  f"nll_f={np.mean(gf):.3f} DnLL={np.mean(gc)-np.mean(gf):+.3f}")
    print("RUNNER_DONE")


if __name__ == "__main__":
    main()
