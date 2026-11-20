"""Keep internal content out of the public result files.

The repository is public, but the generated answers quote internal
accreditation documents and some benchmark questions name staff. Result files
are therefore split in two:

* ``evaluation_results/<name>.json`` -- timings, counts and passage ids only;
  safe to publish.
* ``evaluation_results/private/<name>.json`` -- the full questions and the
  generated answers; git-ignored, stays on this machine.

The question set follows the same rule: ``evaluation/queries.private.json``
(git-ignored) holds the real questions and is used when present;
``evaluation/queries.json`` is the published copy with names replaced.

    .venv\\Scripts\\python.exe -m evaluation.privacy evaluation_results\\pipeline_x.json
    splits an existing result file in place.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation_results"
PRIVATE_DIRECTORY = RESULTS_DIRECTORY / "private"
PUBLIC_QUERIES = PROJECT_ROOT / "evaluation" / "queries.json"
PRIVATE_QUERIES = PROJECT_ROOT / "evaluation" / "queries.private.json"

#: Bracketed citation markers such as [3] or [1][4].
CITATION = re.compile(r"\[(\d+)\]")
#: The fixed sentence the prompt tells the model to use when nothing is found.
NOT_FOUND = "tidak ditemukan pada dokumen yang terindeks"


def queries_path() -> Path:
    """The real question set when available, otherwise the published one."""
    return PRIVATE_QUERIES if PRIVATE_QUERIES.exists() else PUBLIC_QUERIES


def grounding_checks(answer: str, passage_count: int = 6) -> dict:
    """Mechanical checks of one answer that reveal nothing of its content."""
    cited = [int(number) for number in CITATION.findall(answer)]
    return {
        "characters": len(answer),
        "citation_markers": len(cited),
        "distinct_passages_cited": len(set(cited)),
        "invalid_citations": sum(1 for n in cited if n < 1 or n > passage_count),
        "has_citation": bool(cited),
        "says_not_found": NOT_FOUND in answer.lower(),
    }


def split(result: dict) -> tuple[dict, dict]:
    """Return (public, private) versions of a pipeline result."""
    public = copy.deepcopy(result)
    private = {"label": result.get("label"), "created": result.get("created"),
               "queries": result.get("queries"), "answers": []}
    if "queries" in public:
        public["queries"] = [{key: q[key] for key in ("id", "source", "kind") if key in q}
                             for q in public["queries"]]
    for sample in public.get("llm_samples", []):
        answer = sample.pop("answer", None)
        if answer is not None:
            sample.update(grounding_checks(answer))
            private["answers"].append({"query_id": sample.get("query_id"),
                                       "repetition": sample.get("repetition"),
                                       "warmup": sample.get("warmup"), "answer": answer})
    return public, private


def write_split(result: dict, filename: str) -> Path:
    """Write the public file to evaluation_results/ and the private one beside it."""
    public, private = split(result)
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    PRIVATE_DIRECTORY.mkdir(exist_ok=True)
    path = RESULTS_DIRECTORY / filename
    path.write_text(json.dumps(public, indent=1, ensure_ascii=False), encoding="utf-8")
    (PRIVATE_DIRECTORY / filename).write_text(json.dumps(private, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    return path


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        source = Path(argument)
        data = json.loads(source.read_text(encoding="utf-8"))
        if "answers" in data and "stage_summary_ms" not in data:
            continue  # already a private file
        write_split(data, source.name)
        print(f"split {source.name}")
