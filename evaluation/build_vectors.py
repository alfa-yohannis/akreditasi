"""Re-embed the indexed passages with another embedding model.

Reads every passage's text from ``data/index.db`` (read-only) and writes a
vector file in the same format as ``data/vectors.npz`` to
``data/eval/vectors/<model>.npz``, so ``bench_pipeline.py --vectors`` can
search it exactly as the app searches its own index. The source documents on
the shared drive are never touched, and the production index is not changed.

The time and memory it takes are recorded in
``evaluation_results/vectors_<model>_<timestamp>.json``: indexing throughput
is part of the comparison, because a faster model also makes re-indexing
cheaper.

Example::

    .venv\\Scripts\\python.exe -m evaluation.build_vectors intfloat/multilingual-e5-small ^
        --passage-prefix "passage: " --dtype fp16
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SETTINGS  # noqa: E402
from evaluation.bench_pipeline import machine_description, torch_dtype  # noqa: E402

VECTOR_DIRECTORY = PROJECT_ROOT / "data" / "eval" / "vectors"
RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation_results"


def load_passages() -> tuple[np.ndarray, list[str]]:
    """All chunk ids and texts, in chunk-id order, from a read-only connection."""
    connection = sqlite3.connect(f"file:{SETTINGS.database_path.as_posix()}?mode=ro", uri=True)
    rows = connection.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    connection.close()
    return np.array([row[0] for row in rows], dtype=np.int64), [row[1] for row in rows]


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id")
    parser.add_argument("--passage-prefix", default="", help="e.g. 'passage: ' for E5 models")
    parser.add_argument("--passage-prompt-name", default=None,
                        help="named prompt for documents, e.g. 'document' for EmbeddingGemma")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="passages are at most 450 tokens; 512 leaves room for prefixes")
    parser.add_argument("--limit", type=int, default=None, help="encode only the first N passages")
    arguments = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    chunk_ids, texts = load_passages()
    if arguments.limit:
        chunk_ids, texts = chunk_ids[: arguments.limit], texts[: arguments.limit]
    print(f"{len(texts)} passages")

    started = time.perf_counter()
    model = SentenceTransformer(
        arguments.model_id,
        device=arguments.device,
        cache_folder=str(SETTINGS.model_cache_directory),
        model_kwargs={"torch_dtype": torch_dtype(arguments.dtype)},
        trust_remote_code=True,
    )
    model.max_seq_length = arguments.max_seq_length
    load_seconds = time.perf_counter() - started

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    keyword_arguments = {}
    if arguments.passage_prompt_name:
        keyword_arguments["prompt_name"] = arguments.passage_prompt_name
    # Sorting by length lets each batch pad to similar lengths, as the
    # sentence-transformers encoder does internally anyway.
    started = time.perf_counter()
    vectors = model.encode(
        [arguments.passage_prefix + text for text in texts],
        batch_size=arguments.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        **keyword_arguments,
    ).astype(np.float32)
    encode_seconds = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None

    VECTOR_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = VECTOR_DIRECTORY / f"{safe_name(arguments.model_id)}.npz"
    with open(output, "wb") as handle:
        np.savez_compressed(handle, chunk_ids=chunk_ids, vectors=vectors)

    record = {
        "model_id": arguments.model_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": machine_description(),
        "configuration": vars(arguments),
        "passages": len(texts),
        "dimensions": int(vectors.shape[1]),
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "passages_per_second": len(texts) / encode_seconds,
        "peak_gpu_memory_gib": peak_gib,
        "vector_file": str(output.relative_to(PROJECT_ROOT)),
        "vector_file_mib": output.stat().st_size / 2**20,
        "parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIRECTORY / f"vectors_{safe_name(arguments.model_id)}_{stamp}.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("machine", "configuration")},
                     indent=1))


if __name__ == "__main__":
    main()
