# sprag

ReAttention + MAGS on a hybrid (linear-attention + full-attention) language
model. Concrete target: **Qwen3.5-0.8B** (6 full-attention layers + 18 Gated
DeltaNet linear-attention layers, MRoPE with `partial_rotary_factor=0.25`).

The idea:

1. **Inverse RoPE** for the 6 full-attn layers — cache each chunk's
   post-RoPE K/V once at its original document position `A`, and at
   query time apply `R_{B-A}` (a single rotation) to relocate it to
   position `B` in the assembled prompt. No re-encoding required.
2. **LegoLink** for the 18 linear-attn layers — v1 re-forwards the
   retrieved chunks (cheap because the assembled context is short);
   v2 caches the per-chunk Gated DeltaNet state `(G_c, M_c)` so
   `S_after = G_c · S_prefix + M_c` is exact.
3. **MAGS** on the residual stream of layers 11/15/19 — SVD-fit
   an error subspace `B` from `(T+, T-)` activation pairs, and at
   decode time project out the error component when the distance
   exceeds the calibration threshold `τ`.

This repo is the CPU-side scaffolding. Pipeline is built and verified;
GPU is where the actual long-context experiments will run.

## Quick start

```bash
# 1. Sanity: load model, run a short forward
python3 scripts/00_sanity_forward.py

# 2. Unit tests — RoPE numerics + bit-exact identity splice
python3 tests/test_rope.py
python3 tests/test_identity_assembly.py

# 3. End-to-end smoke — build chunk cache, retrieve, generate
python3 scripts/02_smoke_runner.py

# 4. NIAH evaluation
python3 scripts/data/gen_niah.py --out data/niah/niah_4k.jsonl \
    --target_tokens 4096 --n_cases 10 --seed 0
python3 scripts/03_run_niah.py --cases data/niah/niah_4k.jsonl \
    --out data/niah/results_4k.jsonl --modes baseline reattn

# 5. MAGS calibration + full mode
python3 scripts/04_calibrate_mags.py \
    --cases data/niah/niah_4k_calib.jsonl --out data/mags/mags_4k.pkl \
    --n_calib 30 --k_svd 4
python3 scripts/03_run_niah.py --cases data/niah/niah_4k.jsonl \
    --out data/niah/results_full.jsonl \
    --modes baseline reattn full --mags_path data/mags/mags_4k.pkl
```

See **[NOTES.md](NOTES.md)** for the design rationale, what's been verified,
current measurements, and the GPU porting checklist.

## Dependencies

- transformers ≥ 5.3 (needs the `qwen3_5` module)
- torch ≥ 2.5 (bf16 CPU works; bf16/fp16 on GPU)
- safetensors
- The `jinaai/jina-embeddings-v5-text-small` model is loaded with
  `trust_remote_code=True`.

Default model path is hard-coded in `src/sprag/loader.py`. Adjust as needed.
