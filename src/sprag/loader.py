import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def _default_model_path() -> str:
    env = os.environ.get("SPRAG_MODEL_PATH")
    if env:
        return env
    snapshot = "2fc06364715b967f1860aea9cf38778875588b17"
    for root in (
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
    ):
        p = Path(root) / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / snapshot
        if p.exists():
            return str(p)
    return f"~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/{snapshot}"


DEFAULT_MODEL_PATH = _default_model_path()

FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23)
LINEAR_ATTN_LAYERS = tuple(i for i in range(24) if i not in FULL_ATTN_LAYERS)


def load_model(
    model_path: str | Path = None,
    dtype: torch.dtype | None = None,
    attn_impl: str | None = None,
    device: str | torch.device | None = None,
):
    """Load Qwen3.5-0.8B for text-only inference.

    Defaults: on cuda → fp16 + sdpa; on cpu → bf16 + eager (matches CPU
    scaffolding from initial scaffolding session). Override via args or
    env vars SPRAG_MODEL_PATH / SPRAG_DEVICE.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if device is None:
        device = os.environ.get(
            "SPRAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(device)
    on_cuda = device.type == "cuda"

    if dtype is None:
        if on_cuda:
            cap = torch.cuda.get_device_capability(device)
            # bf16 needs Ampere (8.x) or newer; T4 (7.5) → fp16.
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            dtype = torch.bfloat16
    if attn_impl is None:
        attn_impl = "sdpa" if on_cuda else "eager"

    model_path = str(model_path)
    full_cfg = AutoConfig.from_pretrained(model_path)
    text_cfg = full_cfg.text_config
    text_cfg._attn_implementation = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        config=text_cfg,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    if on_cuda:
        # Cheap global knobs that don't affect numerics in our path:
        # - TF32 only applies to the float32 ops left in our pipeline (RoPE
        #   angles, layernorm reductions on some kernels) — speeds them up
        #   without changing the fp16 forward semantics.
        # - matmul precision "high" keeps fp32 accumulators where needed.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cap = torch.cuda.get_device_capability(device)
        if cap[0] < 8:
            # On Turing (T4, cap 7.5) the mem-efficient SDPA backend can't
            # handle SDPA's `enable_gqa=True` (no kernel registered), so
            # transformers' fast path falls back to MATH, which materialises
            # an N×N score matrix (OOM at 16K). Force the slow path —
            # explicit repeat_kv to 8 heads — which mem-efficient handles
            # in ~400 MB at 16K. Patch only the transformers shim, not
            # the SDPA backend enables, so short-seq submodels (e.g. Jina
            # embedder) still get math.
            try:
                from transformers.integrations import sdpa_attention as _sdpa_mod
                _sdpa_mod.use_gqa_in_sdpa = lambda attention_mask, key: False
            except Exception:
                pass

    observed = [
        model.model.layers[i].layer_type for i in range(text_cfg.num_hidden_layers)
    ]
    full_idx = tuple(i for i, t in enumerate(observed) if t == "full_attention")
    assert full_idx == FULL_ATTN_LAYERS, (
        f"layer_types mismatch: expected full at {FULL_ATTN_LAYERS}, got {full_idx}"
    )

    return model, tokenizer, text_cfg
