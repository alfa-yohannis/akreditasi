"""Per-stage retrieval times through the app's own code, as the server runs it.

Times every stage between a question and its six passages -- encoding the
question, search by meaning (and fetching those passages), search by wording,
fusion, reranking, building the prompt -- by wrapping the app's own objects,
so whatever the app does (the reranker's duplicate skipping and length
sorting, the encoder's thread limit) is what is measured. Same protocol as
bench_pipeline.py: warm-up passes discarded, then timed passes over every
question; medians and 95th percentiles in milliseconds.

    .venv\\Scripts\\python.exe -m evaluation.bench_stages --label improved
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402

from app.config import SETTINGS  # noqa: E402
from app.rag import PromptBuilder  # noqa: E402
from app.retrieval import CROSS_ENCODER_RERANKER, QUERY_EMBEDDER, HybridRetriever  # noqa: E402
from app.storage import DOCUMENT_CATALOGUE, VECTOR_INDEX  # noqa: E402
from evaluation.privacy import RESULTS_DIRECTORY, queries_path  # noqa: E402

STAGES = {
    "embed_query": (QUERY_EMBEDDER, "encode_question"),
    "search_by_meaning": (VECTOR_INDEX, "search"),
    "fetch_dense_passages": (DOCUMENT_CATALOGUE, "fetch_passages_by_id"),
    "search_by_wording": (DOCUMENT_CATALOGUE, "search_by_keyword"),
    "rerank": (CROSS_ENCODER_RERANKER, "rerank"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()

    samples: dict[str, list[float]] = {name: [] for name in [*STAGES, "fusion", "build_prompt", "retrieval_total"]}
    recording = {"on": False}

    def wrap(name, owner, attribute):
        original = getattr(owner, attribute)

        def timed(*args, **kwargs):
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                if recording["on"]:
                    samples[name].append((time.perf_counter_ns() - started) / 1e6)

        setattr(owner, attribute, timed)

    for name, (owner, attribute) in STAGES.items():
        wrap(name, owner, attribute)
    retriever = HybridRetriever(SETTINGS, DOCUMENT_CATALOGUE, VECTOR_INDEX, QUERY_EMBEDDER, CROSS_ENCODER_RERANKER)
    wrap("fusion", retriever._fusion, "fuse")
    builder = PromptBuilder(SETTINGS)

    QUERY_EMBEDDER.load()
    CROSS_ENCODER_RERANKER.load()
    questions = [q["text"] for q in json.loads(queries_path().read_text(encoding="utf-8"))["queries"]]
    for pass_number in range(arguments.warmup + arguments.repeats):
        recording["on"] = pass_number >= arguments.warmup
        for question in questions:
            started = time.perf_counter_ns()
            passages = retriever.retrieve(question, None, SETTINGS.answer_passage_count)
            total = (time.perf_counter_ns() - started) / 1e6
            started = time.perf_counter_ns()
            builder.build_system_prompt()
            builder.build_user_prompt(question, passages, [])
            prompt = (time.perf_counter_ns() - started) / 1e6
            if recording["on"]:
                samples["retrieval_total"].append(total)
                samples["build_prompt"].append(prompt)

    summary = {name: {"median": statistics.median(values), "p95": float(np.percentile(values, 95)),
                      "mean": statistics.mean(values), "n": len(values)}
               for name, values in samples.items() if values}
    result = {
        "kind": "stages", "label": arguments.label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": {"query_embedding_device": QUERY_EMBEDDER.device,
                     "query_embedding_backend": QUERY_EMBEDDER.backend_name,
                     "query_embedding_threads": SETTINGS.query_embedding_threads,
                     "reranking_backend": CROSS_ENCODER_RERANKER.backend_name,
                     "rerank_candidates": SETTINGS.reranking_candidate_count,
                     "indexed_passages": VECTOR_INDEX.size,
                     "questions": len(questions), "warmup": arguments.warmup, "repeats": arguments.repeats},
        "stage_summary_ms": summary,
    }
    path = RESULTS_DIRECTORY / f"stages_{arguments.label}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, values in summary.items():
        print(f"{name:<22} median {values['median']:9.1f} ms   p95 {values['p95']:9.1f} ms")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
