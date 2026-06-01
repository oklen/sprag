"""Model loader for the sprag qwen3_5 splice stack.

Originally hardcoded to Qwen3.5-0.8B (24 layers, full-attn at (3,7,11,15,19,23)).
Generalized so the SAME stack works on any qwen3_5 hybrid checkpoint (e.g. the
27B comparison): FULL_ATTN_LAYERS is derived from the model config's
`layer_types` at import time (read from SPRAG_MODEL_PATH, weights not loaded),
and the model is loaded via AutoModelForCausalLM (handles both the 0.8B CausalLM
and the 27B Qwen3_5ForConditionalGeneration → text decoder under model.model).
"""
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _default_model_path() -> str:
    env = os.environ.get("SPRAG_MODEL_PATH")
    if env:
        return env
    snapshot = "2fc06364715b967f1860aea9cf38778875588b17"
    for root in (os.path.expanduser("~/.cache/huggingface/hub"),
                 "/root/.cache/huggingface/hub"):
        p = Path(root) / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / snapshot
        if p.exists():
            return str(p)
    return f"~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/{snapshot}"


DEFAULT_MODEL_PATH = _default_model_path()


def _text_cfg(cfg):
    return getattr(cfg, "text_config", cfg)


def _derive_full_attn_layers(model_path: str):
    """Read config (no weights) and return the tuple of full-attention layer
    indices from `layer_types`. Falls back to the 0.8B hardcoded set."""
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = _text_cfg(cfg)
        lt = getattr(tc, "layer_types", None)
        n = getattr(tc, "num_hidden_layers", 24)
        if lt:
            full = tuple(i for i, t in enumerate(lt) if t == "full_attention")
            if full:
                return full, tuple(i for i in range(n) if i not in set(full))
    except Exception as e:  # offline / config quirk → fall back
        print(f"[loader] could not derive layer_types ({e}); using 0.8B default")
    full = (3, 7, 11, 15, 19, 23)
    return full, tuple(i for i in range(24) if i not in full)


FULL_ATTN_LAYERS, LINEAR_ATTN_LAYERS = _derive_full_attn_layers(DEFAULT_MODEL_PATH)


def load_model(model_path: str | Path = None, dtype=None, attn_impl=None, device=None):
    """Load a qwen3_5 (hybrid) text model for inference.

    Uses AutoModelForCausalLM + device_map='auto' + bf16 by default. The decoder
    layers live at model.model.layers for both the 0.8B CausalLM and the 27B
    multimodal checkpoint (verified). Returns (model, tokenizer, text_config)."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = str(model_path)
    if dtype is None:
        dtype = torch.bfloat16
    if attn_impl is None:
        attn_impl = "sdpa"

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation=attn_impl,
        device_map="auto" if device is None else None, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device is not None:
        model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    text_cfg = _text_cfg(model.config)
    # sanity: observed full-attn layer indices match the derived constant
    try:
        observed = tuple(i for i, l in enumerate(model.model.layers)
                         if getattr(l, "layer_type", None) == "full_attention")
        if observed and observed != FULL_ATTN_LAYERS:
            print(f"[loader] WARN: observed full-attn {observed} != "
                  f"derived {FULL_ATTN_LAYERS}")
    except Exception:
        pass
    return model, tok, text_cfg
