"""Numerical diagnosis of the splice mechanism.

Goal: explain the 47/60 (splice) vs 58/60 (raw text) gap.

Inside each full-attn layer during a sink_oracle_k3 prefill, we capture:
  K_fresh:           freshly-computed key_states[..., b:b+L, :] BEFORE the
                     splice overwrites it (= what the model *would* attend to
                     if we didn't splice)
  K_shifted:         shift_rope(K_cached, delta=b-a)  (= what splice writes in)
  K_cached_no_rot:   K_cached (= what was stored at original position a)

Three quantities of interest:
  (1) rel_err_shifted = ||K_fresh - K_shifted||_F / ||K_fresh||_F
        — the actual splice damage
  (2) rel_err_cached  = ||K_fresh - K_cached_no_rot||_F / ||K_fresh||_F
        — what splice damage would be if RoPE rotation were identity. Tells us
        how much of the divergence is hidden-state drift vs how much is RoPE.
  (3) cos_sim         = mean cosine(K_fresh, K_shifted) per head per token
        — a more interpretable "are these the same direction".

Hypothesis decomposition:
  h_a := hidden_state at position a during the original (full-haystack) cache-build
  h_b := hidden_state at the *same token* now at position b in the assembled prefill
  K_fresh  = RoPE(k_norm(k_proj(h_b)), pos=b)
  K_shifted= RoPE(k_norm(k_proj(h_a)), pos=b)   (= shift_rope(K_cached, b-a))
  K_cached = RoPE(k_norm(k_proj(h_a)), pos=a)

If RoPE math is correct, the only difference between K_fresh and K_shifted is
h_a vs h_b. If RoPE is also buggy, K_shifted will differ from
RoPE(k_norm(k_proj(h_a)), pos=b) computed any other way — but we have no
oracle for that, so we infer it indirectly:
  (1) - (2) ≈ effect of the RoPE delta-rotation
  (2)       ≈ effect of h_a-vs-h_b drift + RoPE position mismatch (the K_cached
              is at position a, not b, so comparing it to K_fresh@b also bakes
              in a position mismatch). For small a, b differences this is
              still a useful proxy.

Run: scripts/13_diagnose_splice.py --suite data/mk/suite_8k --out data/diag/splice_div.json
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

import torch
from safetensors.torch import load_file

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import build_chunk_cache, build_anchor_chunk_cache, load_meta
from sprag.embed import JinaEmbedder
from sprag.assemble import ChunkPlacement, make_inv_freq_for
from sprag.rope import shift_rope
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
from gen_niah import NEEDLES  # type: ignore


def reconstruct_needle(template_id: int, picks: dict) -> str:
    return NEEDLES[template_id][0].format(**picks)


def find_chunk_for_needle(cache_dir: Path, meta, tok, needle_text: str) -> int:
    spine = needle_text[max(0, len(needle_text)//2 - 25): len(needle_text)//2 + 25].lower()
    for c in meta["chunks"]:
        full = tok.decode(load_file(str(cache_dir / f"chunk_{c['id']:05d}.safetensors"))["input_ids"])
        if spine in full.lower():
            return c["id"]
    head = needle_text[:30].lower()
    for c in meta["chunks"]:
        if head in c["text_preview"].lower():
            return c["id"]
    return -1


def _load_chunk(cache_dir: Path, cid: int) -> dict:
    return load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))


def build_placements(cache_dir: Path, chunk_ids: list[int], chunk_lookup: dict,
                      M: int) -> tuple[list[ChunkPlacement], list[int]]:
    """Sink (a=0, b=0, M tokens) + each chunk at sequential b after sink, a from meta."""
    placements = []
    t0 = _load_chunk(cache_dir, 0)
    sink_cached = {li: (t0[f"K_l{li}"][:, :M, :].contiguous(),
                         t0[f"V_l{li}"][:, :M, :].contiguous())
                    for li in FULL_ATTN_LAYERS}
    placements.append(ChunkPlacement(a_start=0, b_start=0, length=M, cached=sink_cached))
    flat = t0["input_ids"][:M].tolist()

    cursor = M
    for cid in chunk_ids:
        t = _load_chunk(cache_dir, cid)
        ids = t["input_ids"]
        L = int(ids.shape[0])
        cached = {li: (t[f"K_l{li}"], t[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(chunk_lookup[cid]["a_start"]),
            b_start=cursor, length=L, cached=cached,
        ))
        flat.extend(ids.tolist())
        cursor += L
    return placements, flat


@torch.no_grad()
def diagnose_one(model, tok, placements, prefix_ids, question, inv_freq):
    """Run one prefill with a capture wrapper. Returns per-layer list of dicts."""
    captures = defaultdict(list)

    # Pre-shift K_cached for each (placement, layer).
    per_layer: dict[int, list] = {li: [] for li in FULL_ATTN_LAYERS}
    for p in placements:
        delta = p.b_start - p.a_start
        for li in FULL_ATTN_LAYERS:
            k_cached, v_cached = p.cached[li]
            k_shift = shift_rope(k_cached.unsqueeze(0), delta, inv_freq).squeeze(0)
            per_layer[li].append((p.b_start, p.length, p.a_start, delta,
                                   k_shift, k_cached, v_cached))

    originals = {}

    def make_forward(attn_module, layer_idx):
        def forward(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kw):
            cfg = attn_module
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, cfg.head_dim)

            query_states, gate = torch.chunk(
                cfg.q_proj(hidden_states).view(*input_shape, -1, cfg.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = cfg.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = cfg.k_norm(cfg.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = cfg.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if key_states.shape[-2] > 1:
                for b_start, L, a_start, delta, k_shift, k_cached, v_cached in per_layer[layer_idx]:
                    K_fresh = key_states[:, :, b_start:b_start+L, :].detach().clone()
                    V_fresh = value_states[:, :, b_start:b_start+L, :].detach().clone()
                    captures[layer_idx].append({
                        "b_start": b_start, "length": L, "a_start": a_start, "delta": delta,
                        "K_fresh": K_fresh.float().cpu(),
                        "K_shifted": k_shift.unsqueeze(0).float().cpu(),
                        "K_cached": k_cached.unsqueeze(0).float().cpu(),
                        "V_fresh": V_fresh.float().cpu(),
                        "V_cached": v_cached.unsqueeze(0).float().cpu(),
                    })
                    key_states[:, :, b_start:b_start+L, :] = k_shift.to(
                        key_states.dtype).to(key_states.device)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, cfg.layer_idx, cache_kwargs
                )

            attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                cfg.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = attn_interface(
                cfg, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=cfg.scaling, **kw,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = cfg.o_proj(attn_output)
            return attn_output, None
        return forward

    try:
        for li in FULL_ATTN_LAYERS:
            attn = model.model.layers[li].self_attn
            originals[li] = attn.forward
            attn.forward = make_forward(attn, li)
        prompt_tail_ids = tok("\n\nQ: " + question + "\nA:",
                               add_special_tokens=False).input_ids
        device = next(model.parameters()).device
        inp = torch.tensor([prefix_ids + prompt_tail_ids], dtype=torch.long, device=device)
        model.model(inp, use_cache=False)
    finally:
        for li, fn in originals.items():
            model.model.layers[li].self_attn.forward = fn

    return captures


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b||_F / ||a||_F over all dims."""
    diff = (a - b).flatten().norm().item()
    base = a.flatten().norm().item()
    return diff / max(base, 1e-12)


def mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity over (head, token) pairs.
    Shapes: (1, n_kv_heads, L, head_dim)."""
    a_flat = a.flatten(0, -2)  # (n_kv * L, head_dim)
    b_flat = b.flatten(0, -2)
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--cache_kind", type=str, default="standard",
                    choices=["standard", "anchor"])
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model).to(device)
    emb = JinaEmbedder()

    # role -> layer -> list of {rel_shifted, rel_cached, cos_shifted, delta}
    rows = []
    template_names = {0: "vault", 1: "secret-keeper", 2: "bookshop"}

    for ci in case_ids:
        cd_src = args.suite / f"case_{ci:02d}"
        haystack = (cd_src / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd_src / "queries.jsonl").open()]

        cache_dir = args.out.parent / f"_diag_case{ci:02d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if args.cache_kind == "anchor":
            build_anchor_chunk_cache(model, tok, haystack, cache_dir,
                                      chunk_size=args.chunk_size, anchor_M=args.M,
                                      embed_fn=emb.encode_passage)
        else:
            build_chunk_cache(model, tok, haystack, cache_dir,
                              chunk_size=args.chunk_size, embed_fn=emb.encode_passage)
        meta = load_meta(cache_dir)
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        for q in queries:
            gold_needle = reconstruct_needle(q["template_id"], q["picks"])
            gold = find_chunk_for_needle(cache_dir, meta, tok, gold_needle)
            if gold < 0:
                continue
            sibs = []
            for q_other in queries:
                if q_other["id"] == q["id"]:
                    continue
                nt = reconstruct_needle(q_other["template_id"], q_other["picks"])
                cid = find_chunk_for_needle(cache_dir, meta, tok, nt)
                if 0 <= cid != gold and cid not in sibs:
                    sibs.append(cid)
                if len(sibs) >= 2:
                    break

            chunk_ids = [gold] + sibs
            placements, flat = build_placements(cache_dir, chunk_ids, chunk_lookup, args.M)
            captures = diagnose_one(model, tok, placements, flat,
                                     q["question"], inv_freq)

            # placements[0] = sink (delta=0), placements[1] = gold, then sibs.
            roles = ["sink", "gold"] + [f"sib{i}" for i in range(len(sibs))]
            for li, plist in captures.items():
                for role, cap in zip(roles, plist):
                    Kf = cap["K_fresh"]
                    Ks = cap["K_shifted"]
                    Kc = cap["K_cached"]
                    Vf = cap["V_fresh"]
                    Vc = cap["V_cached"]
                    rows.append({
                        "case": ci, "qid": q["id"], "template_id": q["template_id"],
                        "template_name": template_names.get(q["template_id"], "?"),
                        "role": role, "layer": li, "delta": cap["delta"],
                        "rel_shifted": relative_l2(Kf, Ks),
                        "rel_cached":  relative_l2(Kf, Kc),
                        "cos_shifted": mean_cos(Kf, Ks),
                        "cos_cached":  mean_cos(Kf, Kc),
                        "rel_v":       relative_l2(Vf, Vc),
                        "cos_v":       mean_cos(Vf, Vc),
                    })
            print(f"  case {ci} q{q['id']} t{q['template_id']} captured "
                  f"{sum(len(v) for v in captures.values())} chunk-layers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(rows, f)

    # Aggregate by layer × role.
    def agg(filter_fn):
        d = defaultdict(list)
        for r in rows:
            if filter_fn(r):
                d[(r["layer"], r["role"])].append(r)
        out = {}
        for (li, role), rs in d.items():
            n = len(rs)
            out[(li, role)] = {
                "n": n,
                "rel_shifted": sum(r["rel_shifted"] for r in rs) / n,
                "rel_cached":  sum(r["rel_cached"]  for r in rs) / n,
                "cos_shifted": sum(r["cos_shifted"] for r in rs) / n,
                "cos_cached":  sum(r["cos_cached"]  for r in rs) / n,
                "rel_v":       sum(r["rel_v"]       for r in rs) / n,
                "cos_v":       sum(r["cos_v"]       for r in rs) / n,
                "delta_mean":  sum(r["delta"]       for r in rs) / n,
            }
        return out

    print("\n=== Splice divergence by (layer, role) — all queries ===")
    print(f"{'layer':>5} {'role':>5} {'n':>4} {'delta':>6} "
          f"{'relS':>7} {'relC':>7} {'cosS':>7} {'cosC':>7} "
          f"{'relV':>7} {'cosV':>7}")
    a = agg(lambda r: True)
    for (li, role) in sorted(a):
        d = a[(li, role)]
        print(f"{li:>5} {role:>5} {d['n']:>4} {d['delta_mean']:>6.0f} "
              f"{d['rel_shifted']:>7.4f} {d['rel_cached']:>7.4f} "
              f"{d['cos_shifted']:>7.4f} {d['cos_cached']:>7.4f} "
              f"{d['rel_v']:>7.4f} {d['cos_v']:>7.4f}")

    print("\n=== Per template (only role=gold) ===")
    for tname in ("vault", "secret-keeper", "bookshop"):
        ag = agg(lambda r, t=tname: r["template_name"] == t and r["role"] == "gold")
        if not ag:
            continue
        print(f"\n  -- {tname} --")
        for (li, role) in sorted(ag):
            d = ag[(li, role)]
            print(f"  layer {li}  n={d['n']}  delta={d['delta_mean']:.0f}  "
                  f"rel_shifted={d['rel_shifted']:.4f}  rel_cached={d['rel_cached']:.4f}  "
                  f"cos_shifted={d['cos_shifted']:.4f}  cos_cached={d['cos_cached']:.4f}")


if __name__ == "__main__":
    main()
