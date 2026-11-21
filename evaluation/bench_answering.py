"""Benchmark the answering model alone, with realistic fresh prompts.

Each prompt is rebuilt from the six passages the baseline run handed to the
model (``retrieval_outputs`` in its result file), so every model and setting
receives exactly the same evidence. Two things make the measurement match what
a user experiences:

* **No prompt cache.** Ollama reuses work for a prompt it has seen before, so
  repeating a prompt makes the first word appear far sooner than it would for
  a new question. Every request therefore starts with a unique marker line,
  which forces the whole prompt to be read again.
* **The app's GPU neighbours.** The app keeps the embedding model on the GPU
  while Ollama answers. ``--resident-embedder cuda`` loads it here the same
  way; ``cpu`` or ``none`` shows what happens when the GPU is left to Ollama.

Shared GPU memory (Windows' overflow of video memory into system RAM) is read
from the "GPU Adapter Memory" performance counters before and after.

    .venv\\Scripts\\python.exe -m evaluation.bench_answering --label qwen3-8b-default ^
        --model qwen3:8b --baseline evaluation_results\\pipeline_baseline_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SETTINGS  # noqa: E402
from app.rag import PromptBuilder  # noqa: E402
from app.storage import DOCUMENT_CATALOGUE  # noqa: E402
from evaluation.bench_pipeline import (  # noqa: E402
    DEFAULT_LLM_QUERY_IDS, OllamaAnswerer, gpu_memory_used_mib, machine_description, summarise)
from evaluation.privacy import queries_path, write_split  # noqa: E402


def adapter_memory_mib() -> dict:
    """Dedicated and shared GPU memory in use, from Windows performance counters."""
    script = ("(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage',"
              "'\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | "
              "ForEach-Object { $_.Path + '=' + [math]::Round($_.CookedValue / 1MB) }")
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    dedicated = shared = 0
    for line in output.splitlines():
        if "=" not in line:
            continue
        path, value = line.rsplit("=", 1)
        if "dedicated usage" in path.lower():
            dedicated = max(dedicated, int(float(value)))
        elif "shared usage" in path.lower():
            shared = max(shared, int(float(value)))
    return {"dedicated_mib": dedicated, "shared_mib": shared}


def unload_all(base_url: str) -> None:
    """Ask an Ollama instance to drop every model it holds, freeing the GPU."""
    import requests

    try:
        loaded = requests.get(f"{base_url}/api/ps", timeout=10).json().get("models", [])
    except Exception:
        return
    for model in loaded:
        requests.post(f"{base_url}/api/generate",
                      json={"model": model["name"], "keep_alive": 0}, timeout=60)
    time.sleep(2)


def build_prompts(baseline_path: Path, query_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Rebuild each question's prompt from the passages the baseline selected."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    questions = {q["id"]: q["text"] for q in
                 json.loads(queries_path().read_text(encoding="utf-8"))["queries"]}
    builder = PromptBuilder(SETTINGS)
    prompts = {}
    for query_id in query_ids:
        final_ids = baseline["retrieval_outputs"][query_id]["final"]
        by_id = DOCUMENT_CATALOGUE.fetch_passages_by_id(final_ids)
        passages = [by_id[chunk_id] for chunk_id in final_ids if chunk_id in by_id]
        for number, passage in enumerate(passages, start=1):
            passage.citation_number = number
        prompts[query_id] = (builder.build_system_prompt(),
                             builder.build_user_prompt(questions[query_id], passages, []))
    return prompts


def load_resident_embedder(device: str):
    """Load the embedding model the way the app does, and keep it in memory."""
    if device == "none":
        return None
    from evaluation.bench_pipeline import QueryEmbedder

    dtype = "fp16" if device == "cuda-fp16" else "fp32"
    device = device.replace("-fp16", "")
    embedder = QueryEmbedder(SETTINGS.embedding_model_name, device, dtype, "", None)
    embedder.encode("pemanasan")  # create the CUDA context and allocate buffers
    return embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True, help="Ollama tag")
    parser.add_argument("--baseline", required=True, help="baseline pipeline result file")
    parser.add_argument("--query-ids", default=DEFAULT_LLM_QUERY_IDS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--resident-embedder", default="cuda", choices=["cuda", "cuda-fp16", "cpu", "none"])
    parser.add_argument("--ollama-url", default=SETTINGS.ollama_base_url)
    parser.add_argument("--other-ollama-url", default=None,
                        help="another Ollama instance to empty first, so models don't share the GPU")
    parser.add_argument("--ollama-env", default="default",
                        help="note of the server settings, e.g. 'OLLAMA_FLASH_ATTENTION=1 ...'")
    parser.add_argument("--num-predict", type=int, default=SETTINGS.maximum_output_tokens)
    parser.add_argument("--keep-alive", default=SETTINGS.ollama_keep_alive)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache-busting", action="store_true",
                        help="repeat prompts verbatim, letting Ollama reuse cached work")
    arguments = parser.parse_args()

    query_ids = [q.strip() for q in arguments.query_ids.split(",") if q.strip()]
    prompts = build_prompts(PROJECT_ROOT / arguments.baseline, query_ids)

    for url in filter(None, [arguments.ollama_url, arguments.other_ollama_url]):
        unload_all(url)
    memory = {"idle": adapter_memory_mib(), "idle_nvidia_smi_mib": gpu_memory_used_mib()}
    embedder = load_resident_embedder(arguments.resident_embedder)
    memory["with_embedder"] = adapter_memory_mib()

    answerer = OllamaAnswerer(arguments.model, arguments.seed, arguments.ollama_url,
                              arguments.num_predict, arguments.keep_alive)
    print(f"[{arguments.label}] {arguments.model} via {arguments.ollama_url}, "
          f"embedder on {arguments.resident_embedder}, {len(query_ids)} questions, "
          f"warm-up {arguments.warmup}, repeats {arguments.repeats}")

    samples, residency = [], None
    for pass_number in range(arguments.warmup + arguments.repeats):
        warmup = pass_number < arguments.warmup
        for query_id in query_ids:
            system_prompt, user_prompt = prompts[query_id]
            if not arguments.no_cache_busting:
                system_prompt = f"[permintaan {uuid.uuid4().hex[:12]}]\n{system_prompt}"
            sample = answerer.answer(system_prompt, user_prompt)
            sample.update({"query_id": query_id, "warmup": warmup,
                           "repetition": pass_number - arguments.warmup})
            samples.append(sample)
            if residency is None:
                residency = answerer.residency()
                memory["with_model"] = adapter_memory_mib()
                memory["with_model_nvidia_smi_mib"] = gpu_memory_used_mib()
            print(f"  {query_id} pass {pass_number + 1}: first word "
                  f"{(sample['first_token_ms'] or 0) / 1000:.1f} s, total {sample['total_ms'] / 1000:.1f} s, "
                  f"{sample['output_tokens']} tokens at {sample['output_tokens_per_s'] or 0:.1f}/s, "
                  f"prompt {sample['prompt_tokens_per_s'] or 0:.0f}/s{' (warm-up)' if warmup else ''}",
                  flush=True)

    timed = [s for s in samples if not s["warmup"]]
    summary = {}
    for key in ["first_token_ms", "total_ms", "prompt_eval_ms", "generation_ms", "prompt_tokens",
                "output_tokens", "prompt_tokens_per_s", "output_tokens_per_s", "load_ms"]:
        values = [s[key] for s in timed if s.get(key) is not None]
        if values:
            summary[key] = summarise(values)

    result = {
        "kind": "answering",
        "label": arguments.label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": machine_description(),
        "configuration": vars(arguments),
        "settings": {"num_ctx": SETTINGS.ollama_context_window,
                     "temperature": SETTINGS.generation_temperature,
                     "num_predict": arguments.num_predict},
        "gpu_memory": memory,
        "ollama_residency": residency,
        "llm_summary": summary,
        "llm_samples": samples,
        "queries": [{"id": q} for q in query_ids],
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = write_split(result, f"answering_{arguments.label}_{stamp}.json")
    del embedder

    print(f"\n{arguments.label}: {len(timed)} answers; residency {residency}; memory {memory}")
    print(f"{'measure':<22}{'mean':>10}{'median':>10}{'p95':>10}")
    for name, s in summary.items():
        print(f"{name:<22}{s['mean']:>10.1f}{s['median']:>10.1f}{s['p95']:>10.1f}")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
