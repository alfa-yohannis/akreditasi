"""Check that the figures an answer quotes are in the passages it was given.

The prompt tells the model to quote numbers, names and decree numbers exactly,
and only from the six passages. This checks the part of that rule a program
can check without labelled answers: every *figure* in an answer -- a number of
two or more digits (years, counts, percentages, page numbers), a decimal, or a
code mixing letters and digits (course codes, decree numbers) -- must appear in
the six passages, their document names and locations, or the question itself.

A figure that appears in none of them was invented, miscopied or computed by
the model. Computed figures (a total the documents do not state) are not
always wrong, but the system's rules forbid them as well, so they count.

What it cannot see: a correct figure attached to the wrong subject, and
claims without figures. It is a floor on errors, not a measure of correctness.

    .venv\\Scripts\\python.exe -m evaluation.faithfulness [result files...]

Without arguments it checks every ``answering_*`` and ``endtoend_*`` result.
Counts are written to ``evaluation_results/faithfulness_<stamp>.json``; the
unsupported figures themselves, which quote internal documents, go to the
git-ignored ``evaluation_results/private/`` twin.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.storage import DOCUMENT_CATALOGUE  # noqa: E402
from evaluation.privacy import CITATION, PRIVATE_DIRECTORY, RESULTS_DIRECTORY, queries_path  # noqa: E402

BASELINE = RESULTS_DIRECTORY / "pipeline_baseline_20261120-164319.json"

#: Codes mixing letters and digits: IT140904, SK-045/XYZ/VI/2025, A3.
CODE = re.compile(r"\b(?=[A-Za-z0-9./-]*\d)(?=[A-Za-z0-9./-]*[A-Za-z])[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*\b")
#: Numbers: 12, 2025, 12,5, 1.497, 3.14 -- with Indonesian or English separators.
NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")
#: "1." / "2)" at the start of a line: list numbering, not a figure.
LIST_NUMBER = re.compile(r"(?m)^\s*\d+[.)]\s")


def normalise_number(token: str) -> str:
    """One spelling per number: separators dropped, so 1.497 = 1,497 = 1497."""
    return re.sub(r"[.,]", "", token).lstrip("0") or "0"


#: Extracted PDF text often spaces out separators: "149 /SK-R/PRADITA".
SPACED_SEPARATOR = re.compile(r"\s*([/\-])\s*")


def figures(text: str) -> tuple[set[str], set[str]]:
    """The numbers (normalised) and codes (lower-case) quoted in a text."""
    text = CITATION.sub(" ", LIST_NUMBER.sub(" ", text))
    text = SPACED_SEPARATOR.sub(r"\1", text)
    codes = {match.lower() for match in CODE.findall(text)}
    numbers = set()
    for match in NUMBER.findall(text):
        digits = re.sub(r"[.,]", "", match)
        if len(digits) >= 2 or re.search(r"[.,]", match):
            numbers.add(normalise_number(match))
    return numbers, codes


def reference_text(question: str, passages) -> str:
    """Everything the model was shown: the question and the six passages."""
    parts = [question]
    for passage in passages:
        parts += [passage.document_name, passage.section_label or "", passage.text]
    return "\n".join(parts)


def unsupported(answer: str, reference: str) -> list[str]:
    """Figures in the answer that the reference does not contain."""
    answer_numbers, answer_codes = figures(answer)
    reference_numbers, reference_codes = figures(reference)
    flat_reference = SPACED_SEPARATOR.sub(r"\1", reference.lower())
    missing = []
    for number in sorted(answer_numbers):
        if number not in reference_numbers:
            missing.append(number)
    for code in sorted(answer_codes):
        if code not in reference_codes and code not in flat_reference:
            missing.append(code)
    return missing


def checkable_count(answer: str) -> int:
    numbers, codes = figures(answer)
    return len(numbers) + len(codes)


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or sorted(
        p for pattern in ("answering_answer-*.json", "endtoend_*.json")
        for p in RESULTS_DIRECTORY.glob(pattern))
    questions = {q["id"]: q["text"] for q in
                 json.loads(queries_path().read_text(encoding="utf-8"))["queries"]}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    references: dict[str, str] = {}

    def reference_for(query_id: str) -> str:
        if query_id not in references:
            final_ids = baseline["retrieval_outputs"][query_id]["final"]
            by_id = DOCUMENT_CATALOGUE.fetch_passages_by_id(final_ids)
            passages = [by_id[i] for i in final_ids if i in by_id]
            references[query_id] = reference_text(questions[query_id], passages)
        return references[query_id]

    public, private = {"created": datetime.now().isoformat(timespec="seconds"),
                       "method": __doc__.split("\n\n")[1].replace("\n", " "), "runs": {}}, {"runs": {}}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        private_twin = PRIVATE_DIRECTORY / path.name
        if not private_twin.exists():
            continue
        answers = json.loads(private_twin.read_text(encoding="utf-8"))["answers"]
        label = result["label"]
        # End-to-end runs retrieve afresh; they read the same passages only if
        # retrieval matches the baseline, which overlap@6 shows is mostly so.
        per_answer = []
        for entry in answers:
            if entry["warmup"]:
                continue
            reference = reference_for(entry["query_id"])
            missing = unsupported(entry["answer"], reference)
            per_answer.append({"query_id": entry["query_id"], "repetition": entry["repetition"],
                               "figures": checkable_count(entry["answer"]),
                               "unsupported": len(missing), "_missing": missing})
        total_figures = sum(a["figures"] for a in per_answer)
        total_missing = sum(a["unsupported"] for a in per_answer)
        by_question = {}
        for a in per_answer:
            q = by_question.setdefault(a["query_id"], {"answers": 0, "figures": 0, "unsupported": 0,
                                                       "answers_with_unsupported": 0})
            q["answers"] += 1
            q["figures"] += a["figures"]
            q["unsupported"] += a["unsupported"]
            q["answers_with_unsupported"] += bool(a["unsupported"])
        public["runs"][label] = {
            "file": path.name, "answers": len(per_answer), "figures": total_figures,
            "unsupported": total_missing,
            "unsupported_share": total_missing / total_figures if total_figures else None,
            "answers_with_unsupported": sum(bool(a["unsupported"]) for a in per_answer),
            "by_question": by_question,
        }
        # Only the aggregates above are public; the figures themselves stay private.
        private["runs"][label] = [a for a in per_answer if a["unsupported"]]
        print(f"{label:<30} answers {len(per_answer):>3}  figures {total_figures:>4}  "
              f"unsupported {total_missing:>3}  answers with any {public['runs'][label]['answers_with_unsupported']:>2}"
              f"  {by_question and {q: (v['figures'], v['unsupported']) for q, v in by_question.items()}}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"faithfulness_{stamp}.json"
    (RESULTS_DIRECTORY / name).write_text(json.dumps(public, indent=1, ensure_ascii=False), encoding="utf-8")
    PRIVATE_DIRECTORY.mkdir(exist_ok=True)
    (PRIVATE_DIRECTORY / name).write_text(json.dumps(private, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote evaluation_results/{name}")


if __name__ == "__main__":
    main()
