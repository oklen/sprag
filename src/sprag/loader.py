from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)

FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23)
LINEAR_ATTN_LAYERS = tuple(i for i in range(24) if i not in FULL_ATTN_LAYERS)


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "eager",
):
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
    model.eval()

    observed = [
        model.model.layers[i].layer_type for i in range(text_cfg.num_hidden_layers)
    ]
    full_idx = tuple(i for i, t in enumerate(observed) if t == "full_attention")
    assert full_idx == FULL_ATTN_LAYERS, (
        f"layer_types mismatch: expected full at {FULL_ATTN_LAYERS}, got {full_idx}"
    )

    return model, tokenizer, text_cfg
