"""Per-stage latency benchmark of the question-answering pipeline.

Times every stage a question passes through, using the app's own components
wherever possible so the numbers describe the real system:

    embed question -> search by meaning -> fetch passages      (dense path)
                   -> search by wording (FTS5, joins included)  (sparse path)
    -> reciprocal rank fusion -> rerank -> build prompt -> answering model

Each configuration is first warmed up (models loaded, CUDA kernels compiled,
caches filled) and then timed over N repetitions of every query. Results,
including every raw sample and the passages each stage returned, are written
to ``evaluation_results/pipeline_<label>_<timestamp>.json``.

Examples (from the project root)::

    .venv\\Scripts\\python.exe -m evaluation.bench_pipeline --label baseline
    .venv\\Scripts\\python.exe -m evaluation.bench_pipeline --label rerank-gpu-fp16 ^
        --rerank-device cuda --rerank-dtype fp16 --llm none
    .venv\\Scripts\\python.exe -m evaluation.bench_pipeline --label e5-small ^
        --embedder intfloat/multilingual-e5-small --query-prefix "query: " ^
        --vectors data/eval/vectors/intfloat__multilingual-e5-small.npz --llm none

The answering model is called through Ollama's HTTP API with exactly the
options the app sends (see ``app/language_models.py``), plus a fixed seed so
repeated runs generate comparable answers.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SETTINGS  # noqa: E402
from app.rag import PromptBuilder  # noqa: E402
from app.retrieval import ReciprocalRankFusion  # noqa: E402
from app.storage import DOCUMENT_CATALOGUE  # noqa: E402

RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation_results"
from evaluation.privacy import queries_path, write_split  # noqa: E402

#: Questions sent to the answering model, one of each common kind: a table
#: lookup, a person, a prose list, the curriculum and an identifier. Five
#: rather than all twenty keeps 3 + 10 passes to about half an hour per model.
DEFAULT_LLM_QUERY_IDS = "q01,q04,q09,q12,q17"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def milliseconds_since(start_ns: int) -> float:
    """Elapsed time since ``start_ns`` (a perf_counter_ns reading), in ms."""
    return (time.perf_counter_ns() - start_ns) / 1e6


def summarise(samples: list[float]) -> dict:
    """Mean, median, p95, min, max and standard deviation of ``samples``."""
    if not samples:
        return {}
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def gpu_memory_used_mib() -> int | None:
    """Total GPU memory in use according to nvidia-smi, across all processes."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(output.strip().splitlines()[0])
    except Exception:
        return None


def machine_description() -> dict:
    """Hardware and library versions, recorded with every result file."""
    import torch
    import sentence_transformers
    import transformers

    description = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        description["gpu"] = torch.cuda.get_device_name(0)
        description["gpu_memory_gib"] = round(properties.total_memory / 2**30, 2)
    try:
        import psutil

        description["ram_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
        description["cpu_cores_logical"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    return description


def torch_dtype(name: str):
    """Map 'fp32' / 'fp16' / 'bf16' to a torch dtype."""
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


# ---------------------------------------------------------------------------
# Pipeline components under test
# ---------------------------------------------------------------------------

class QueryEmbedder:
    """Encodes questions the way a given embedding model expects.

    E5-style models need a text prefix ("query: "); Qwen3-Embedding and
    EmbeddingGemma use a named prompt. BGE-M3, the baseline, needs neither.
    """

    def __init__(self, model_id: str, device: str, dtype: str,
                 query_prefix: str, query_prompt_name: str | None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.query_prefix = query_prefix
        self.query_prompt_name = query_prompt_name
        self.model = SentenceTransformer(
            model_id,
            device=device,
            cache_folder=str(SETTINGS.model_cache_directory),
            model_kwargs={"torch_dtype": torch_dtype(dtype)},
            trust_remote_code=True,
        )

    def encode(self, question: str) -> np.ndarray:
        """One normalised float32 vector for ``question``."""
        keyword_arguments = {}
        if self.query_prompt_name:
            keyword_arguments["prompt_name"] = self.query_prompt_name
        vector = self.model.encode(
            [self.query_prefix + question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **keyword_arguments,
        )
        return np.asarray(vector, dtype=np.float32)[0]


class ExactVectorSearch:
    """The app's search by meaning: one dot product over every stored vector.

    The maths is copied from ``app.storage.VectorIndex.search`` so alternative
    vector files (built for other embedding models) are searched identically.
    """

    def __init__(self, archive_path: Path) -> None:
        with np.load(archive_path) as archive:
            self.chunk_ids = archive["chunk_ids"].astype(np.int64)
            self.vectors = archive["vectors"].astype(np.float32)

    def search(self, question_vector: np.ndarray, limit: int) -> list[tuple[int, float]]:
        similarities = self.vectors @ question_vector.astype(np.float32)
        count = min(limit, similarities.shape[0])
        top = np.argpartition(-similarities, count - 1)[:count]
        top = top[np.argsort(-similarities[top])]
        return [(int(self.chunk_ids[i]), float(similarities[i])) for i in top]


class PassageReranker:
    """Cross-encoder re-scoring, as in ``app.reranking.CrossEncoderReranker``."""

    def __init__(self, model_id: str, device: str, dtype: str,
                 batch_size: int, max_length: int) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_id,
            device=device,
            max_length=max_length,
            cache_folder=str(SETTINGS.model_cache_directory),
            model_kwargs={"torch_dtype": torch_dtype(dtype)},
            trust_remote_code=True,
        )

    def rerank(self, question: str, passages: list, keep: int) -> list:
        pairs = [(question, passage.text) for passage in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip(passages, scores), key=lambda pair: float(pair[1]), reverse=True)
        kept = []
        for passage, score in ranked[:keep]:
            passage.reranked_score = float(score)
            kept.append(passage)
        return kept


class OllamaAnswerer:
    """Streams an answer from Ollama and records where the time went."""

    def __init__(self, model_tag: str, seed: int, base_url: str, num_predict: int,
                 keep_alive: str) -> None:
        self.model_tag = model_tag
        self.seed = seed
        self.base_url = base_url
        self.num_predict = num_predict
        self.keep_alive = keep_alive

    def answer(self, system_prompt: str, user_prompt: str, attempts: int = 3) -> dict:
        """One timed answer, retried if Ollama fails before answering.

        Ollama 0.40 occasionally answers HTTP 500 when its own connection to
        the model server has gone idle ("server closed idle connection"); the
        model is fine and the next request succeeds. Such a request is retried
        from scratch, so its timing is a clean one, and the retry is recorded.
        """
        import requests

        for attempt in range(1, attempts + 1):
            try:
                result = self._answer_once(system_prompt, user_prompt)
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status != 500 or attempt == attempts:
                    raise
                print(f"  Ollama HTTP 500, retrying ({attempt}/{attempts - 1})", flush=True)
                time.sleep(2)
                continue
            result["retries"] = attempt - 1
            return result
        raise AssertionError("unreachable")

    def _answer_once(self, system_prompt: str, user_prompt: str) -> dict:
        import requests

        payload = {
            "model": self.model_tag,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": SETTINGS.generation_temperature,
                "num_ctx": SETTINGS.ollama_context_window,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        started = time.perf_counter_ns()
        first_token_ms = None
        text_parts: list[str] = []
        final_event: dict = {}
        with requests.post(f"{self.base_url}/api/chat", json=payload,
                           stream=True, timeout=(10, 600)) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                event = json.loads(raw_line)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                content = (event.get("message") or {}).get("content", "")
                if content:
                    if first_token_ms is None:
                        first_token_ms = milliseconds_since(started)
                    text_parts.append(content)
                if event.get("done"):
                    final_event = event
        total_ms = milliseconds_since(started)
        nanoseconds = 1e6
        eval_count = final_event.get("eval_count", 0)
        eval_ms = final_event.get("eval_duration", 0) / nanoseconds
        prompt_count = final_event.get("prompt_eval_count", 0)
        prompt_ms = final_event.get("prompt_eval_duration", 0) / nanoseconds
        return {
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "load_ms": final_event.get("load_duration", 0) / nanoseconds,
            "prompt_tokens": prompt_count,
            "prompt_eval_ms": prompt_ms,
            "prompt_tokens_per_s": prompt_count / (prompt_ms / 1000) if prompt_ms else None,
            "output_tokens": eval_count,
            "generation_ms": eval_ms,
            "output_tokens_per_s": eval_count / (eval_ms / 1000) if eval_ms else None,
            "done_reason": final_event.get("done_reason"),
            "answer": "".join(text_parts),
        }

    def residency(self) -> dict | None:
        """How much of the loaded model sits in VRAM, from Ollama's /api/ps."""
        import requests

        try:
            models = requests.get(f"{self.base_url}/api/ps", timeout=10).json()["models"]
        except Exception:
            return None
        for model in models:
            if model.get("name") == self.model_tag or model.get("model") == self.model_tag:
                size, in_vram = model.get("size", 0), model.get("size_vram", 0)
                return {
                    "size_mib": round(size / 2**20),
                    "vram_mib": round(in_vram / 2**20),
                    "share_on_gpu": round(in_vram / size, 3) if size else None,
                    "context_length": model.get("context_length"),
                }
        return None


# ---------------------------------------------------------------------------
# One pass of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """Timings (ms) and outputs of one query through retrieval."""

    query_id: str
    repetition: int
    warmup: bool
    timings: dict = field(default_factory=dict)
    dense_ids: list[int] = field(default_factory=list)
    sparse_ids: list[int] = field(default_factory=list)
    fused_ids: list[int] = field(default_factory=list)
    final_ids: list[int] = field(default_factory=list)


class PipelineUnderTest:
    """Wires the configured components into the app's retrieval sequence."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.fusion = ReciprocalRankFusion(SETTINGS.fusion_constant)
        self.prompt_builder = PromptBuilder(SETTINGS)

        started = time.perf_counter_ns()
        self.vector_search = ExactVectorSearch(PROJECT_ROOT / arguments.vectors)
        self.load_ms = {"vectors": milliseconds_since(started)}

        started = time.perf_counter_ns()
        self.embedder = QueryEmbedder(arguments.embedder, arguments.embed_device,
                                      arguments.embed_dtype, arguments.query_prefix,
                                      arguments.query_prompt_name)
        self.load_ms["embedder"] = milliseconds_since(started)

        self.reranker = None
        if arguments.reranker.lower() != "none":
            started = time.perf_counter_ns()
            if arguments.reranker_backend == "openvino":
                from evaluation.openvino_reranker import OPENVINO_DIRECTORY, OpenVinoReranker

                self.reranker = OpenVinoReranker(OPENVINO_DIRECTORY / arguments.openvino_variant,
                                                 arguments.rerank_batch, arguments.rerank_max_length,
                                                 arguments.openvino_hint)
            else:
                self.reranker = PassageReranker(arguments.reranker, arguments.rerank_device,
                                                arguments.rerank_dtype, arguments.rerank_batch,
                                                arguments.rerank_max_length)
            self.load_ms["reranker"] = milliseconds_since(started)

    def retrieve(self, query_id: str, question: str, repetition: int, warmup: bool):
        """Run retrieval once, timing each stage. Returns (record, passages)."""
        record = StageRecord(query_id, repetition, warmup)
        timings = record.timings
        total_started = time.perf_counter_ns()

        started = time.perf_counter_ns()
        question_vector = self.embedder.encode(question)
        timings["embed_query"] = milliseconds_since(started)

        started = time.perf_counter_ns()
        scored_ids = self.vector_search.search(question_vector, self.arguments.dense_candidates)
        timings["search_by_meaning"] = milliseconds_since(started)

        started = time.perf_counter_ns()
        passages_by_id = DOCUMENT_CATALOGUE.fetch_passages_by_id([cid for cid, _ in scored_ids])
        dense_passages = []
        for chunk_id, similarity in scored_ids:
            passage = passages_by_id.get(chunk_id)
            if passage is not None:
                passage.dense_score = similarity
                dense_passages.append(passage)
        timings["fetch_dense_passages"] = milliseconds_since(started)

        started = time.perf_counter_ns()
        sparse_passages = DOCUMENT_CATALOGUE.search_by_keyword(
            question, self.arguments.sparse_candidates, None)
        timings["search_by_wording"] = milliseconds_since(started)

        started = time.perf_counter_ns()
        fused = self.fusion.fuse([dense_passages, sparse_passages])
        candidates = fused[: self.arguments.rerank_candidates]
        timings["fusion"] = milliseconds_since(started)

        started = time.perf_counter_ns()
        if self.reranker is not None:
            final = self.reranker.rerank(question, candidates, SETTINGS.answer_passage_count)
        else:
            final = candidates[: SETTINGS.answer_passage_count]
        timings["rerank"] = milliseconds_since(started)
        for number, passage in enumerate(final, start=1):
            passage.citation_number = number

        started = time.perf_counter_ns()
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(question, final, [])
        timings["build_prompt"] = milliseconds_since(started)

        timings["both_searches"] = timings["search_by_meaning"] + timings["fetch_dense_passages"] \
            + timings["search_by_wording"]
        timings["retrieval_total"] = milliseconds_since(total_started)

        record.dense_ids = [p.chunk_id for p in dense_passages]
        record.sparse_ids = [p.chunk_id for p in sparse_passages]
        record.fused_ids = [p.chunk_id for p in candidates]
        record.final_ids = [p.chunk_id for p in final]
        return record, (system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="name for this configuration")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up passes over all queries")
    parser.add_argument("--repeats", type=int, default=10, help="timed passes over all queries")
    parser.add_argument("--embedder", default=SETTINGS.embedding_model_name)
    parser.add_argument("--embed-device", default="cuda")
    parser.add_argument("--embed-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--query-prefix", default="", help="e.g. 'query: ' for E5 models")
    parser.add_argument("--query-prompt-name", default=None, help="e.g. 'query' for Qwen3-Embedding")
    parser.add_argument("--vectors", default="data/vectors.npz",
                        help="vector file matching --embedder, relative to the project root")
    parser.add_argument("--reranker", default=SETTINGS.reranking_model_name, help="'none' to skip")
    parser.add_argument("--rerank-device", default=SETTINGS.reranking_device)
    parser.add_argument("--rerank-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--dense-candidates", type=int, default=SETTINGS.dense_candidate_count,
                        help="passages nominated by search by meaning")
    parser.add_argument("--sparse-candidates", type=int, default=SETTINGS.sparse_candidate_count,
                        help="passages nominated by search by wording")
    parser.add_argument("--rerank-candidates", type=int, default=SETTINGS.reranking_candidate_count)
    parser.add_argument("--rerank-batch", type=int, default=8)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--reranker-backend", default="torch", choices=["torch", "openvino"],
                        help="openvino runs the same model converted by evaluation.openvino_reranker")
    parser.add_argument("--openvino-variant", default="int8", choices=["fp32", "int8"])
    parser.add_argument("--openvino-hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT"])
    parser.add_argument("--llm", default=SETTINGS.ollama_model_name, help="Ollama tag, or 'none'")
    parser.add_argument("--llm-query-ids", default=DEFAULT_LLM_QUERY_IDS)
    parser.add_argument("--llm-warmup", type=int, default=3,
                        help="warm-up passes over the answering-model questions")
    parser.add_argument("--llm-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Ollama sampling seed")
    parser.add_argument("--ollama-url", default=SETTINGS.ollama_base_url,
                        help="a separate Ollama instance, e.g. one started with tuned settings")
    parser.add_argument("--num-predict", type=int, default=SETTINGS.maximum_output_tokens,
                        help="cap on answer length in tokens")
    parser.add_argument("--keep-alive", default=SETTINGS.ollama_keep_alive)
    parser.add_argument("--ollama-env", default="",
                        help="free-text note of the Ollama server settings, recorded with the results")
    parser.add_argument("--save-query-vectors", default=None,
                        help="write the question vectors (queries.json order) to this .npy path")
    return parser.parse_args()


def run(arguments: argparse.Namespace) -> Path:
    queries = json.loads(queries_path().read_text(encoding="utf-8"))["queries"]
    print(f"[{arguments.label}] {len(queries)} queries, warm-up {arguments.warmup}, "
          f"repeats {arguments.repeats}")
    gpu_before_mib = gpu_memory_used_mib()

    pipeline = PipelineUnderTest(arguments)
    print(f"  loaded in {pipeline.load_ms}")

    records: list[StageRecord] = []
    prompts: dict[str, tuple[str, str]] = {}
    for pass_number in range(arguments.warmup + arguments.repeats):
        warmup = pass_number < arguments.warmup
        repetition = pass_number - arguments.warmup
        for query in queries:
            record, prompt = pipeline.retrieve(query["id"], query["text"], repetition, warmup)
            records.append(record)
            prompts[query["id"]] = prompt
        print(f"  pass {pass_number + 1}/{arguments.warmup + arguments.repeats}"
              f"{' (warm-up)' if warmup else ''} done")

    gpu_after_retrieval_mib = gpu_memory_used_mib()
    if arguments.save_query_vectors:
        vectors = np.stack([pipeline.embedder.encode(q["text"]) for q in queries])
        np.save(PROJECT_ROOT / arguments.save_query_vectors, vectors)
        print(f"  saved query vectors {vectors.shape} to {arguments.save_query_vectors}")

    llm_samples: list[dict] = []
    residency = None
    if arguments.llm.lower() != "none":
        answerer = OllamaAnswerer(arguments.llm, arguments.seed, arguments.ollama_url,
                                  arguments.num_predict, arguments.keep_alive)
        llm_ids = [qid.strip() for qid in arguments.llm_query_ids.split(",") if qid.strip()]
        for pass_number in range(arguments.llm_warmup + arguments.llm_repeats):
            warmup = pass_number < arguments.llm_warmup
            for query_id in llm_ids:
                system_prompt, user_prompt = prompts[query_id]
                sample = answerer.answer(system_prompt, user_prompt)
                sample.update({"query_id": query_id, "warmup": warmup,
                               "repetition": pass_number - arguments.llm_warmup})
                llm_samples.append(sample)
                if residency is None:
                    residency = answerer.residency()
                print(f"  llm {arguments.llm} {query_id} pass {pass_number + 1}: "
                      f"first token {sample['first_token_ms'] or 0:.0f} ms, "
                      f"total {sample['total_ms']:.0f} ms, {sample['output_tokens']} tokens"
                      f"{' (warm-up)' if warmup else ''}")

    return write_results(arguments, queries, pipeline, records, llm_samples, residency,
                         gpu_before_mib, gpu_after_retrieval_mib)


def write_results(arguments, queries, pipeline, records, llm_samples, residency,
                  gpu_before_mib, gpu_after_retrieval_mib) -> Path:
    timed = [record for record in records if not record.warmup]
    stage_names = list(timed[0].timings.keys()) if timed else []
    stage_summary = {name: summarise([r.timings[name] for r in timed]) for name in stage_names}

    timed_llm = [sample for sample in llm_samples if not sample["warmup"]]
    llm_summary = {}
    for key in ["first_token_ms", "total_ms", "prompt_eval_ms", "generation_ms",
                "prompt_tokens", "output_tokens", "prompt_tokens_per_s", "output_tokens_per_s",
                "load_ms"]:
        values = [s[key] for s in timed_llm if s.get(key) is not None]
        if values:
            llm_summary[key] = summarise(values)

    # The passages each query ended up with, from the first timed pass.
    first_pass = {r.query_id: r for r in timed if r.repetition == 0}
    retrieval_outputs = {
        qid: {"dense": r.dense_ids, "sparse": r.sparse_ids, "fused": r.fused_ids,
              "final": r.final_ids}
        for qid, r in first_pass.items()
    }

    result = {
        "label": arguments.label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": machine_description(),
        "configuration": vars(arguments),
        "settings": {
            "dense_candidates": arguments.dense_candidates,
            "sparse_candidates": arguments.sparse_candidates,
            "rerank_candidates": arguments.rerank_candidates,
            "answer_passages": SETTINGS.answer_passage_count,
            "rrf_k": SETTINGS.fusion_constant,
            "ollama_num_ctx": SETTINGS.ollama_context_window,
            "max_output_tokens": arguments.num_predict,
            "temperature": SETTINGS.generation_temperature,
        },
        "load_ms": pipeline.load_ms,
        "gpu_memory_mib": {"before": gpu_before_mib, "after_retrieval": gpu_after_retrieval_mib},
        "ollama_residency": residency,
        "stage_summary_ms": stage_summary,
        "llm_summary": llm_summary,
        "retrieval_outputs": retrieval_outputs,
        "stage_samples": [r.__dict__ for r in records],
        "llm_samples": llm_samples,
        "queries": queries,
    }
    # Answers and question texts go to the git-ignored private/ folder.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = write_split(result, f"pipeline_{arguments.label}_{stamp}.json")

    print(f"\n{arguments.label}: per-query stage latency (ms), {len(timed)} samples")
    print(f"{'stage':<22}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for name, s in stage_summary.items():
        print(f"{name:<22}{s['mean']:>9.2f}{s['median']:>9.2f}{s['p95']:>9.2f}{s['max']:>9.2f}")
    if llm_summary:
        print(f"\nanswering model {arguments.llm}, {len(timed_llm)} answers; residency {residency}")
        for name, s in llm_summary.items():
            print(f"{name:<22}{s['mean']:>9.1f}{s['median']:>9.1f}{s['p95']:>9.1f}{s['max']:>9.1f}")
    print(f"\nwrote {path.relative_to(PROJECT_ROOT)}")
    return path


if __name__ == "__main__":
    run(parse_arguments())
