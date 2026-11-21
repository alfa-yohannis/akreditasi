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

    .venv\\Scripts\\python.exe -m evaluation.privacy --regrade evaluation_results\\answering_x.json
    recomputes the grounding checks of a public file from its private answers.
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


#: A refusal opens with the not-found sentence; later in an answer, the same
#: words only flag one missing detail ("... rincian indikatornya tidak
#: ditemukan pada dokumen yang terindeks").
REFUSAL_WITHIN_CHARACTERS = 80


def grounding_checks(answer: str, passage_count: int = 6) -> dict:
    """Mechanical checks of one answer that reveal nothing of its content.

    ``says_not_found`` is the raw phrase match. It is split into ``refuses``
    (the answer opens with the not-found sentence, or cites nothing) and
    ``notes_missing_detail`` (a cited answer that adds the sentence about one
    sub-question), because the two mean opposite things for grounding.
    """
    cited = [int(number) for number in CITATION.findall(answer)]
    position = answer.lower().find(NOT_FOUND)
    says_not_found = position >= 0
    refuses = says_not_found and (position < REFUSAL_WITHIN_CHARACTERS or not cited)
    return {
        "characters": len(answer),
        "citation_markers": len(cited),
        "distinct_passages_cited": len(set(cited)),
        "invalid_citations": sum(1 for n in cited if n < 1 or n > passage_count),
        "has_citation": bool(cited),
        "says_not_found": says_not_found,
        "refuses": refuses,
        "notes_missing_detail": says_not_found and not refuses,
    }


def regrade(public_path: Path) -> int:
    """Recompute the grounding checks of a public file from its private twin."""
    private_path = PRIVATE_DIRECTORY / public_path.name
    if not private_path.exists():
        return 0
    public = json.loads(public_path.read_text(encoding="utf-8"))
    answers = json.loads(private_path.read_text(encoding="utf-8"))["answers"]
    samples = public.get("llm_samples", [])
    if len(samples) != len(answers):
        raise ValueError(f"{public_path.name}: {len(samples)} samples, {len(answers)} answers")
    for sample, entry in zip(samples, answers):
        if (sample.get("query_id"), sample.get("repetition")) != (entry["query_id"], entry["repetition"]):
            raise ValueError(f"{public_path.name}: samples and answers are out of step")
        sample.update(grounding_checks(entry["answer"]))
    public_path.write_text(json.dumps(public, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(samples)


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
    if sys.argv[1:2] == ["--regrade"]:
        for argument in sys.argv[2:]:
            print(f"regraded {regrade(Path(argument))} answers in {Path(argument).name}")
        sys.exit(0)
    for argument in sys.argv[1:]:
        source = Path(argument)
        data = json.loads(source.read_text(encoding="utf-8"))
        if "answers" in data and "stage_summary_ms" not in data:
            continue  # already a private file
        write_split(data, source.name)
        print(f"split {source.name}")
