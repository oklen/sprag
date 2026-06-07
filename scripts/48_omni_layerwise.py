"""Check #2: layer-wise cross-modal recovery on Qwen3-Omni Thinker.
xrecover @ cov100 (all video kept, audio DROPPED at use). cached = video KV
prebaked WITH audio (carries audio trace); fresh = video re-encoded WITHOUT audio.
Both at identical kept positions (pos_matched). Sweep cumulative depth d: reuse
cached KV for layers [0..d), fresh for [d..L). gold-option NLL(d). ΔNLL(d) =
nll(d) - nll_fresh; fraction recovered = ΔNLL(d)/(nll_cached - nll_fresh).
Built-in identity: d=0 == fresh, d=L == cached. Tells WHICH layers hold the trace.
"""
import os, sys, json, argparse, importlib.util
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
# env defaults for the imported 44 module globals (read at import time)
os.environ.setdefault("DATASET", "videomme")
os.environ.setdefault("DROP_AUDIO_AT_USE", "1")
os.environ.setdefault("EGO_VIDEO_DIR", "/tmp/videomme")
os.environ.setdefault("EGO_META", "/home/tiger/videomme_meta")
_spec = importlib.util.spec_from_file_location("omni44", str(ROOT / "scripts/44_omni_coverage.py"))
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)
ok = m.omni_kv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--depth_step", type=int, default=4)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.n_frames % 2: args.n_frames += 1
    eng = m.Engine(); dev = eng.dev
    df = m.load_meta(); vmap = m.build_video_map(m.VIDEO_DIR)
    done = {}
    if os.path.exists(args.out):
        done = {r["uid"]: r for r in json.load(open(args.out))}
    rows_out = list(done.values())
    gi = -1; n = 0
    for _, row in df.iterrows():
        uid, vidkey, question, options, gold = m.parse_row(row)
        gi += 1
        if (gi % args.num_shards) != args.shard_id: continue
        if uid in done: continue
        vid = vmap.get(str(vidkey))
        if not vid: continue
        try:
            frames = m.extract_frames(vid, args.n_frames, m.FRAME_DIR)
            wav = os.path.join(m.FRAME_DIR, f"a_{args.shard_id}.wav")
            audio = m.extract_audio(vid, wav)
            if audio is None: continue
            inpA = eng.build(frames, question, audio=audio)
            posA = eng.rope(inpA); fullA = eng.prefill(eng.fwd_kwargs(inpA), posA)
            idrow = inpA["input_ids"][0].tolist(); T = len(idrow)
            span = ok.video_token_span(idrow, eng.VID)
            t_grid = int(inpA["video_grid_thw"][0][0].item())
            ranges = ok.tgroup_ranges(span, t_grid)
            adrop = {eng.AUD, eng.AUD_S, eng.AUD_E}
            audio_pos = set(i for i, t in enumerate(idrow) if t in adrop)
            groups = ok.select_coverage_groups(t_grid, 1.0, mode="uniform")  # cov100
            keep = ok.build_keep_idx(T, span, ranges, groups, dev)
            keep = torch.tensor([i for i in keep.tolist() if i not in audio_pos], dtype=torch.long, device=dev)
            keep_max = int(posA[:, 0, :][:, keep].max().item())
            cached_g = ok.gather_kv(fullA, keep)
            kept_frames = []
            for g in groups: kept_frames += [frames[2 * g], frames[2 * g + 1]]
            inpF = eng.build(kept_frames, question, audio=None)
            fids = inpF["input_ids"][0].tolist()
            cached_ids = [idrow[i] for i in keep.tolist()]
            if fids != cached_ids:
                continue  # not pos-matched -> can't layer-mix
            fpos = posA[:, :, keep]
            freshF = eng.prefill(eng.fwd_kwargs(inpF), fpos)
            cl = ok.cache_layers(cached_g); fl = ok.cache_layers(freshF)
            nL = len(cl)
            gopt = [" " + options[gold]]
            def gnll(layers):
                c = ok.build_cache_from_layers(layers)
                return ok.mc_option_nll(eng.thinker, eng.tok, c, keep_max, "\n", gopt, dev)[0]
            nll_fresh = gnll(fl); nll_cached = gnll(cl)
            depths = list(range(0, nL + 1, args.depth_step))
            if nL not in depths: depths.append(nL)
            curve = {}
            for d in depths:
                curve[d] = gnll([cl[li] if li < d else fl[li] for li in range(nL)])
            rec = {"uid": str(uid), "nL": nL, "nll_fresh": nll_fresh, "nll_cached": nll_cached,
                   "delta_full": nll_cached - nll_fresh, "curve": curve}
            rows_out.append(rec)
            n += 1
            json.dump(rows_out, open(args.out + ".tmp", "w")); os.replace(args.out + ".tmp", args.out)
            if n % 3 == 0:
                print(f"[{n}] uid={uid} d0={curve[0]:.3f}(=fresh {nll_fresh:.3f}) "
                      f"dL={curve[nL]:.3f}(=cached {nll_cached:.3f}) delta={nll_cached-nll_fresh:+.3f}", flush=True)
            if n >= args.limit: break
        except Exception as e:
            print(f"skip uid={uid}: {type(e).__name__} {e}", flush=True)
            continue
    print(f"DONE shard {args.shard_id}: {n} records -> {args.out}")

if __name__ == "__main__":
    main()
