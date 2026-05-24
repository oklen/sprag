"""Sanity check: load Qwen3.5-0.8B text-only and run a short forward + generate."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model, FULL_ATTN_LAYERS, LINEAR_ATTN_LAYERS


def main():
    print("Loading model...")
    t0 = time.time()
    model, tok, cfg = load_model()
    print(f"Loaded in {time.time()-t0:.1f}s; dtype={next(model.parameters()).dtype}")
    print(f"Full-attn layers: {FULL_ATTN_LAYERS}")
    print(f"Linear-attn layers: {LINEAR_ATTN_LAYERS}")
    print(f"hidden_size={cfg.hidden_size}  head_dim={cfg.head_dim}  "
          f"n_q={cfg.num_attention_heads}  n_kv={cfg.num_key_value_heads}  "
          f"partial_rotary={cfg.rope_parameters.get('partial_rotary_factor', 1.0)}")

    prompt = "The capital of France is"
    inputs = tok(prompt, return_tensors="pt")
    print(f"\nPrompt: {prompt!r}  ({inputs.input_ids.shape[1]} tokens)")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generated ({dt:.1f}s): {text!r}")

    print("\n[forward shape check] forward(input_ids) on 32-token prompt")
    ids = tok("Hello world, this is a sanity check on the hidden state shapes.",
              return_tensors="pt").input_ids
    with torch.no_grad():
        h = model.model(ids).last_hidden_state
    print(f"  last_hidden_state: {tuple(h.shape)}  dtype={h.dtype}")


if __name__ == "__main__":
    main()
