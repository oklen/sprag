"""Build the chunk cache on a small synthetic doc as a smoke test."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model
from sprag.chunk_cache import build_chunk_cache
from sprag.embed import JinaEmbedder


SAMPLE_DOC = (
    "The history of computing has roots in attempts to mechanize counting. "
    "In 1822 Charles Babbage proposed the Difference Engine. " * 20
    + "Halfway through the document, a magic phrase appears: "
    "the secret number is forty-two and the keeper is Octavia. "
    "After the magic phrase, more historical context follows. "
    + "Ada Lovelace wrote what is considered the first computer program. " * 20
    + "Alan Turing later formalized the notion of an algorithm. " * 20
)


def main():
    print("Loading Qwen3.5...")
    model, tok, cfg = load_model()
    print("Loading Jina embedder...")
    emb = JinaEmbedder()

    out_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "smoke"
    t0 = time.time()
    chunks, meta = build_chunk_cache(
        model, tok, SAMPLE_DOC, out_dir,
        chunk_size=128,
        embed_fn=emb.encode_passage,
    )
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"meta: tokens={meta['num_tokens']}, chunks={meta['num_chunks']}")
    print(f"Chunk 0 preview: {meta['chunks'][0]['text_preview']!r}")

    # Inspect tensors in chunk_0
    from sprag.chunk_cache import load_chunk
    c0 = load_chunk(out_dir, 0)
    print("\nChunk 0 tensors:")
    for k, v in c0.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
