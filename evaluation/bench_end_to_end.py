"""End-to-end benchmark through the app's own answering code.

Sends each question through ``RagEngine.stream_answer`` -- the code path the
web interface uses -- and times what a user experiences:

* **cards**: until the ``sources`` event (the evidence cards appear);
* **first word**: until the first ``token`` event;
* **done**: until the answer is complete.

A configuration is a set of the app's own settings, passed as ``--set
NAME=VALUE`` and applied to the environment *before* the app is imported,
exactly as ``.env`` would apply them. Each configuration runs in its own
process. Every request gets a unique first line in the system prompt, so the
answering model cannot reuse cached work from an earlier identical prompt --
every real question is new, too.

    .venv\\Scripts\\python.exe -m evaluation.bench_end_to_end --label current ^
        --set RERANK_BACKEND=torch --set QUERY_EMBED_DEVICE=cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="an app setting, as it would appear in .env")
    parser.add_argument("--query-ids", default="q01,q04,q09,q12,q17")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


ARGUMENTS = parse_arguments()
for assignment in ARGUMENTS.set:
    name, _, value = assignment.partition("=")
    os.environ[name.strip()] = value.strip()

# Imported only now, so the settings above take effect.
from app.config import SETTINGS  # noqa: E402
from app.language_models import LanguageModelFactory  # noqa: E402
from app.rag import PromptBuilder, QuestionRequest, RagEngine  # noqa: E402
from app.retrieval import CROSS_ENCODER_RERANKER, QUERY_EMBEDDER, HybridRetriever  # noqa: E402
from app.storage import DOCUMENT_CATALOGUE, VECTOR_INDEX  # noqa: E402
from evaluation.bench_pipeline import gpu_memory_used_mib, machine_description, summarise  # noqa: E402
from evaluation.privacy import queries_path, write_split  # noqa: E402


class FreshPromptBuilder(PromptBuilder):
    """The app's prompt builder, with a unique first line per request."""

    def build_system_prompt(self) -> str:
        return f"[permintaan {uuid.uuid4().hex[:12]}]\n{super().build_system_prompt()}"


def ask(engine: RagEngine, question: str) -> dict:
    """Stream one answer and time its milestones (ms from the request)."""
    started = time.perf_counter_ns()
    since = lambda: (time.perf_counter_ns() - started) / 1e6  # noqa: E731
    timings = {"cards_ms": None, "first_token_ms": None, "done_ms": None}
    answer_parts, error = [], None
    for event in engine.stream_answer(QuestionRequest(question=question)):
        if event.name == "sources" and timings["cards_ms"] is None:
            timings["cards_ms"] = since()
        elif event.name == "token":
            if timings["first_token_ms"] is None:
                timings["first_token_ms"] = since()
            answer_parts.append(event.payload.get("text", ""))
        elif event.name == "error":
            error = event.payload.get("message")
        elif event.name == "done":
            timings["done_ms"] = since()
    timings["answer"] = "".join(answer_parts)
    timings["error"] = error
    return timings


def main() -> None:
    queries = {q["id"]: q["text"] for q in
               json.loads(queries_path().read_text(encoding="utf-8"))["queries"]}
    query_ids = [q.strip() for q in ARGUMENTS.query_ids.split(",") if q.strip()]

    retriever = HybridRetriever(SETTINGS, DOCUMENT_CATALOGUE, VECTOR_INDEX,
                                QUERY_EMBEDDER, CROSS_ENCODER_RERANKER)
    engine = RagEngine(SETTINGS, retriever, LanguageModelFactory(SETTINGS))
    engine._prompt_builder = FreshPromptBuilder(SETTINGS)

    print(f"[{ARGUMENTS.label}] settings {ARGUMENTS.set}; model {SETTINGS.ollama_model_name} "
          f"at {SETTINGS.ollama_base_url}", flush=True)
    samples = []
    for pass_number in range(ARGUMENTS.warmup + ARGUMENTS.repeats):
        warmup = pass_number < ARGUMENTS.warmup
        for query_id in query_ids:
            sample = ask(engine, queries[query_id])
            sample.update({"query_id": query_id, "warmup": warmup,
                           "repetition": pass_number - ARGUMENTS.warmup})
            samples.append(sample)
            print(f"  {query_id} pass {pass_number + 1}: cards {sample['cards_ms'] / 1000:.1f} s, "
                  f"first word {(sample['first_token_ms'] or 0) / 1000:.1f} s, "
                  f"done {sample['done_ms'] / 1000:.1f} s{' (warm-up)' if warmup else ''}"
                  f"{' ERROR ' + sample['error'] if sample['error'] else ''}", flush=True)

    timed = [s for s in samples if not s["warmup"]]
    summary = {key: summarise([s[key] for s in timed if s[key] is not None])
               for key in ("cards_ms", "first_token_ms", "done_ms")}
    result = {
        "kind": "end_to_end",
        "label": ARGUMENTS.label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": machine_description(),
        "configuration": {"settings": ARGUMENTS.set, "query_ids": query_ids,
                          "warmup": ARGUMENTS.warmup, "repeats": ARGUMENTS.repeats},
        "resolved": {
            "reranking_backend": CROSS_ENCODER_RERANKER.backend_name,
            "query_embedding_device": QUERY_EMBEDDER.device,
            "ollama_model": SETTINGS.ollama_model_name,
            "ollama_url": SETTINGS.ollama_base_url,
        },
        "gpu_memory_used_mib_at_end": gpu_memory_used_mib(),
        "summary_ms": summary,
        # "answer" texts are moved to evaluation_results/private by write_split
        "llm_samples": samples,
        "queries": [{"id": q} for q in query_ids],
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = write_split(result, f"endtoend_{ARGUMENTS.label}_{stamp}.json")
    print(f"\n{ARGUMENTS.label}: {len(timed)} answers; resolved {result['resolved']}")
    for name, s in summary.items():
        print(f"{name:<16}{s['mean']:>10.0f}{s['median']:>10.0f}{s['p95']:>10.0f}")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
