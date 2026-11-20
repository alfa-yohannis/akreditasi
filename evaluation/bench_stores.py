r"""Benchmark alternative implementations of the RAG system's two search stores.

The application keeps two search stores (see ``app/storage.py``):

* a **vector index** - L2-normalised BGE-M3 chunk embeddings in
  ``data/vectors.npz``, searched exactly with a NumPy dot product followed by
  ``argpartition`` (``VectorIndex.search``);
* a **keyword index** - the SQLite FTS5 table ``chunk_search`` in
  ``data/index.db``, ranked with ``bm25()``
  (``DocumentCatalogue.search_by_keyword``).

This script times alternatives for each store and measures how closely their
top-40 results agree with the application's.

Vector methods (top 40; a query is one 1024-d unit vector)
    NumPy float32 exact (baseline), NumPy float16 exact, FAISS IndexFlatIP,
    FAISS HNSW, usearch HNSW, and FAISS 8-bit scalar / 1-bit binary
    quantisation followed by an exact float32 rescoring pass.

Keyword methods (top 40; a query is the question text)
    FTS5 with the app's SQL (baseline), FTS5 returning rowids only (no joins),
    FTS5 ranking first and joining only the top 40, bm25s (FTS5's formula and
    the library default), and tantivy.

Methodology
-----------
* Every method is built once from the same in-memory data. "Build" is the
  time to turn that data into a searchable index. For the float32 baseline it
  is the time to load ``vectors.npz`` the way ``VectorIndex.load`` does; for
  FTS5 it is the time to fill a fresh copy of the table under
  ``data/eval/stores``. Writing an index to disk is timed separately as
  "save", except for FTS5 and tantivy, which write while they build.
* A timed call covers the whole per-question path. For keyword methods this
  includes turning the question into a query with ``KeywordQueryBuilder``.
* Each method gets ``--warmup`` untimed passes over all queries, then
  ``--repeats`` timed passes. Every call is timed on its own with
  ``time.perf_counter_ns``. Statistics are taken over all
  repeats x queries samples, and the raw samples are kept in the JSON output.
* Agreement with the baseline is ``|method top-k & baseline top-k| /
  |baseline top-k|``, averaged over queries. It is called recall@k for vector
  search, where the baseline is exact, and overlap@k for keyword search, where
  the baseline is just another BM25 implementation.
* The corpus holds many duplicate chunks (the same file in several folders),
  so the baseline's k-th score is often shared by more than k results and
  which of them make the cut is arbitrary. A *tie-aware* variant is therefore
  reported as well: a returned chunk counts when its score *on the baseline's
  scale* (exact float32 similarity, or FTS5 ``bm25()``) is at least the
  baseline's k-th score.
* Thread pools are left at library defaults and recorded.

Keyword tokenisation
--------------------
FTS5's ``unicode61 remove_diacritics 2`` tokeniser treats runs of letters and
digits as tokens (``_`` and punctuation separate them), folds case and strips
diacritics. bm25s is fed tokens produced the same way (see
:func:`fts5_like_tokens`), and tantivy uses its simple tokenizer with
lowercase and ASCII-folding filters. FTS5 indexes every token, stop words
included, and stop words are removed only from the query. The alternatives
therefore index every token too, and take their query terms from the same
``KeywordQueryBuilder`` expression, so all engines see the same stop words.

Safety
------
``data/index.db`` is opened read-only (``mode=ro``) and ``data/vectors.npz`` is
only read. ``app.storage`` is deliberately *not* imported, because importing it
opens ``index.db`` for writing. The query builder and the SQL are copied here
verbatim instead, and checked against ``app/storage.py`` at start-up. Every
index this script builds is written under ``data/eval/stores/``.

Usage
-----
    .venv\Scripts\python.exe evaluation\bench_stores.py
        [--warmup 3] [--repeats 10] [--query-vectors question_vectors.npy]
        [--only vector|keyword]

``--query-vectors`` takes a ``.npy`` array of shape (20, 1024) holding the
embeddings of the questions in ``evaluation/queries.json``, in file order.
Without it, 20 stored passage vectors stand in for the questions.
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

# --- Paths --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# So `from evaluation.privacy import ...` works when run as a plain script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATABASE_PATH = PROJECT_ROOT / "data" / "index.db"
VECTOR_ARCHIVE_PATH = PROJECT_ROOT / "data" / "vectors.npz"
STORES_DIRECTORY = PROJECT_ROOT / "data" / "eval" / "stores"
QUERIES_PATH = PROJECT_ROOT / "evaluation" / "queries.json"
RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation_results"
APP_STORAGE_SOURCE = PROJECT_ROOT / "app" / "storage.py"

# --- Benchmark parameters -----------------------------------------------------

#: Candidates each store returns per question (DENSE_/SPARSE_CANDIDATES).
TOP_K = 40
#: Passages finally shown to the language model (TOP_K in app/config.py).
SMALL_K = 6
#: HNSW graph degree ("M") for both FAISS and usearch.
HNSW_CONNECTIVITY = 32
#: HNSW construction beam, the same for both libraries (usearch's default)
#: so that the two graphs are comparable.
HNSW_CONSTRUCTION_EXPANSION = 128
#: HNSW search beams. A beam narrower than k is meaningless for a top-40
#: search, so the sweep starts at 40.
HNSW_SEARCH_EXPANSIONS = (40, 64, 128, 256)
#: Candidates a quantised index hands to the exact float32 rescoring pass.
RESCORE_DEPTHS = (100, 400)
#: Seed for choosing stand-in query vectors when none are supplied.
STAND_IN_SEED = 2026
#: Two results tie when their baseline scores differ by less than this.
TIE_TOLERANCE = 1e-6
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10

#: Optional libraries: label -> module imported. Missing ones are skipped.
OPTIONAL_LIBRARIES = {
    "faiss": "faiss",
    "usearch": "usearch.index",
    "bm25s": "bm25s",
    "tantivy": "tantivy",
}
#: Distributions whose versions are recorded in the output.
RECORDED_DISTRIBUTIONS = (
    "numpy", "scipy", "faiss-cpu", "usearch", "bm25s", "tantivy", "threadpoolctl",
)
#: Environment variables that cap library thread pools; recorded, never set.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)

# --- Copied from app/storage.py (verified at start-up) ------------------------
# app.storage cannot be imported: doing so opens data/index.db read-write and
# commits a schema write. The class below and the SQL strings are verbatim
# copies, whitespace included, and verify_app_copies() checks that they still
# match the application's source.


class KeywordQueryBuilder:
    """Converts a natural-language question into a safe FTS5 MATCH expression.

    FTS5 treats characters such as ``"``, ``*``, ``(`` and ``-`` as operators, so
    a raw question routinely produces a syntax error. This builder keeps only
    word characters, drops very short and very common Indonesian stop words, and
    joins the remainder with ``OR`` so partial matches still rank.
    """

    _WORD_PATTERN = re.compile(r"[\wÀ-ɏ]+", re.UNICODE)

    #: High-frequency Indonesian and English function words that add no signal.
    STOP_WORDS = frozenset({
        "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah",
        "atau", "ini", "itu", "dalam", "oleh", "sebagai", "juga", "akan", "ada",
        "tidak", "bisa", "dapat", "apa", "siapa", "bagaimana", "berapa", "kapan",
        "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "is", "are",
        "what", "how", "who", "when", "which",
    })

    #: Words shorter than this carry too little signal to be worth matching.
    MINIMUM_WORD_LENGTH = 3

    def build(self, question: str) -> str:
        """Return an FTS5 expression for ``question``, or "" when unusable."""
        candidate_words = self._WORD_PATTERN.findall(question.lower())
        meaningful_words = [
            word
            for word in candidate_words
            if len(word) >= self.MINIMUM_WORD_LENGTH and word not in self.STOP_WORDS
        ]
        if not meaningful_words:
            # Every word was filtered out; fall back to the raw words so a
            # question like "apa itu EWMP" still matches on "EWMP".
            meaningful_words = [word for word in candidate_words if len(word) > 1]
        if not meaningful_words:
            return ""
        # Quoting each term makes it a literal, neutralising FTS5 operators.
        return " OR ".join(f'"{word}"' for word in dict.fromkeys(meaningful_words))


#: The SELECT of DocumentCatalogue.search_by_keyword, copied verbatim.
APP_KEYWORD_SELECT = """
            SELECT c.chunk_id, c.document_id, c.section_label, c.text, c.produced_by_ocr,
                   d.display_name, d.relative_path, d.absolute_path,
                   bm25(chunk_search) AS relevance
            FROM chunk_search
            JOIN chunks    c ON c.chunk_id = chunk_search.rowid
            JOIN documents d ON d.document_id = c.document_id
            WHERE chunk_search MATCH ?
        """
#: The suffix the app appends when no folder filter is set.
APP_KEYWORD_ORDER = " ORDER BY relevance LIMIT ?"
APP_KEYWORD_SQL = APP_KEYWORD_SELECT + APP_KEYWORD_ORDER

#: The same ranking without the joins that fetch text and document metadata.
ROWID_ONLY_SQL = (
    "SELECT rowid, bm25(chunk_search) AS relevance FROM chunk_search "
    "WHERE chunk_search MATCH ? ORDER BY relevance LIMIT ?"
)
#: The app's columns, but ranked first and joined for the top rows only. The
#: app's SQL joins every matching row before sorting. Without a folder filter
#: both return the same rows; with one, the filter would have to move inside.
RANK_FIRST_SQL = """
    SELECT c.chunk_id, c.document_id, c.section_label, c.text, c.produced_by_ocr,
           d.display_name, d.relative_path, d.absolute_path, ranked.relevance
    FROM (SELECT rowid, bm25(chunk_search) AS relevance FROM chunk_search
          WHERE chunk_search MATCH ? ORDER BY relevance LIMIT ?) AS ranked
    JOIN chunks    c ON c.chunk_id = ranked.rowid
    JOIN documents d ON d.document_id = c.document_id
    ORDER BY ranked.relevance
"""
#: FTS5 scores for given rowids. bm25() uses whole-table statistics, so a
#: row's score does not depend on the rowid filter.
FTS5_SCORE_SQL = (
    "SELECT rowid, bm25(chunk_search) FROM chunk_search "
    "WHERE chunk_search MATCH ? AND rowid IN ({placeholders})"
)
#: The app's FTS5 table definition, used to time a fresh build.
FTS5_CREATE_SQL = (
    "CREATE VIRTUAL TABLE chunk_search "
    "USING fts5(text, tokenize='unicode61 remove_diacritics 2')"
)

# --- Keyword tokenisation -----------------------------------------------------

#: Runs of letters and digits. "_" and punctuation separate tokens, as in unicode61.
FTS5_TOKEN_PATTERN = re.compile(r"[^\W_]+")
#: One quoted term of a KeywordQueryBuilder expression.
QUOTED_TERM_PATTERN = re.compile(r'"([^"]*)"')
#: bm25s scoring variants: (label, constructor arguments).
BM25S_VARIANTS = (
    ("FTS5 formula", {"method": "robertson", "k1": 1.2, "b": 0.75}),
    ("library default", {"method": "lucene", "k1": 1.5, "b": 0.75}),
)
#: Name under which the FTS5-like analyzer is registered in tantivy.
TANTIVY_TOKENIZER = "fts5_like"


# --- Data structures ----------------------------------------------------------


@dataclass
class SearchMethod:
    """A built index plus the function that answers one query with chunk ids."""

    name: str
    search: Callable[[Any], list[int]]
    build_seconds: float
    index_bytes: int | None
    size_note: str
    save_seconds: float | None = None
    settings: dict = field(default_factory=dict)
    #: Called once before the method is measured (e.g. to set a search beam).
    prepare: Callable[[], None] | None = None


@dataclass
class VectorCorpus:
    """The stored passage embeddings, as VectorIndex holds them."""

    chunk_ids: np.ndarray
    vectors: np.ndarray
    load_seconds: float


@dataclass
class TokenisedCorpus:
    """Chunk texts tokenised FTS5-style and mapped to integer token ids."""

    chunk_ids: np.ndarray
    token_ids: list[list[int]]
    vocabulary: dict[str, int]
    seconds: float


#: Scores chunks on the baseline's scale for one query; higher is better.
Scorer = Callable[[int, list[int]], dict[int, float]]


@dataclass
class Reference:
    """The baseline's top results per query, and its scores for any chunk."""

    agreement_name: str
    top_ids: list[list[int]]
    scorer: Scorer
    top_scores: list[dict[int, float]]


def make_reference(agreement_name: str, top_ids: list[list[int]], scorer: Scorer) -> Reference:
    """Score the baseline's own results once, for the tie-aware comparison."""
    top_scores = [scorer(query_index, ids) for query_index, ids in enumerate(top_ids)]
    return Reference(agreement_name, top_ids, scorer, top_scores)


#: A labelled function that builds one or more search methods.
Builder = tuple[str, Callable[[], list[SearchMethod]]]


# --- Entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the requested benchmarks, write the JSON report, print a summary."""
    arguments = parse_arguments(argv)
    started_at = datetime.now()
    queries = load_queries()
    modules, unavailable = import_optional_libraries()
    STORES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    notices: list[str] = []
    app_copies = verify_app_copies()
    if not all(app_copies.values()):
        report_notice(notices, "WARNING: the copied KeywordQueryBuilder/SQL differ from "
                      f"app/storage.py ({app_copies}); results may not reflect the app.")
    faiss = modules.get("faiss")
    if faiss is not None and "AVX" not in faiss.get_compile_options().upper():
        report_notice(notices, "NOTE: FAISS loaded its generic build without AVX2/AVX-512 "
                      f"(compile options: {faiss.get_compile_options()!r}); FAISS timings "
                      "understate a SIMD-optimised build.")

    report: dict[str, Any] = {
        "benchmark": "search stores",
        "started_at": started_at.isoformat(timespec="seconds"),
        "parameters": describe_parameters(arguments, len(queries)),
        "machine": describe_machine(),
        "libraries": library_versions(),
        "unavailable_libraries": unavailable,
        "app_copies_match": app_copies,
        "notices": notices,
    }
    if arguments.only in (None, "vector"):
        report["vector"] = run_vector_benchmark(queries, arguments, modules, unavailable, notices)
    if arguments.only in (None, "keyword"):
        report["keyword"] = run_keyword_benchmark(queries, arguments, modules, unavailable)
    report["runtime"] = describe_runtime(faiss)
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    output_path = write_report(report, started_at)
    print_summary(report)
    print(f"\nResults written to {output_path}")
    return 0


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(
        description="Benchmark alternatives for the RAG system's vector and keyword stores."
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"untimed passes over all queries per method (default {DEFAULT_WARMUP})")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"timed passes over all queries per method (default {DEFAULT_REPEATS})")
    parser.add_argument("--query-vectors", type=Path, default=None,
                        help="(queries x 1024) .npy of question embeddings, in queries.json order; "
                             "without it, stored passage vectors stand in")
    parser.add_argument("--only", choices=("vector", "keyword"), default=None,
                        help="run only one of the two benchmarks")
    arguments = parser.parse_args(argv)
    if arguments.warmup < 0:
        parser.error("--warmup must be 0 or more")
    if arguments.repeats < 1:
        parser.error("--repeats must be 1 or more")
    return arguments


def load_queries() -> list[dict]:
    """Return the benchmark questions: the git-ignored real set when present."""
    from evaluation.privacy import queries_path

    with queries_path().open(encoding="utf-8") as handle:
        return json.load(handle)["queries"]


def import_optional_libraries() -> tuple[dict[str, Any], dict[str, str]]:
    """Import each optional library, recording why any of them is unavailable."""
    modules: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for label, module_name in OPTIONAL_LIBRARIES.items():
        try:
            modules[label] = importlib.import_module(module_name)
        except Exception as error:  # ImportError, or a DLL load failure on Windows
            unavailable[label] = f"{type(error).__name__}: {error}"
            print(f"NOTE: {label} is unavailable; its methods are skipped ({unavailable[label]})")
    return modules, unavailable


def report_notice(notices: list[str], message: str) -> None:
    """Print a notice and keep it for the JSON report."""
    print(message)
    notices.append(message)


def verify_app_copies() -> dict[str, bool]:
    """Check the copied query builder and SQL against app/storage.py."""
    source = APP_STORAGE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    result = {"keyword_query_builder": False, "keyword_sql": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KeywordQueryBuilder":
            app_class = ast.get_source_segment(source, node) or ""
            result["keyword_query_builder"] = (
                app_class.strip() == inspect.getsource(KeywordQueryBuilder).strip()
            )
        if isinstance(node, ast.FunctionDef) and node.name == "search_by_keyword":
            strings = {
                constant.value
                for constant in ast.walk(node)
                if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
            }
            result["keyword_sql"] = APP_KEYWORD_SELECT in strings and APP_KEYWORD_ORDER in strings
    return result


# --- Shared measurement -------------------------------------------------------


def measure_builders(
    builders: list[Builder],
    queries: list,
    reference: Reference,
    arguments: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    """Build and measure every method; failures are recorded, not raised."""
    records: list[dict] = []
    failures: list[dict] = []
    for label, build in builders:
        print(f"- building {label} ...", flush=True)
        try:
            methods = build()
        except Exception as error:
            failures.append(failure_entry(label, "build", error))
            continue
        for method in methods:
            try:
                record = measure_method(method, queries, arguments.warmup, arguments.repeats)
            except Exception as error:
                failures.append(failure_entry(method.name, "search", error))
                continue
            record.update(compare_with_reference(record["top_ids"], reference))
            records.append(record)
            print_progress_line(record, reference.agreement_name)
        del methods
        gc.collect()
    return records, failures


def measure_method(method: SearchMethod, queries: list, warmup: int, repeats: int) -> dict:
    """Collect one method's results, then warm it up and time it."""
    if method.prepare is not None:
        method.prepare()
    top_ids = [method.search(query) for query in queries]
    run_passes(method.search, queries, warmup)
    gc.collect()
    samples_ns = time_repetitions(method.search, queries, repeats)
    return {
        "method": method.name,
        "settings": method.settings,
        "build_seconds": method.build_seconds,
        "save_seconds": method.save_seconds,
        "index_bytes": method.index_bytes,
        "size_note": method.size_note,
        "latency_ms": summarise_latency(samples_ns),
        "result_counts": [len(ids) for ids in top_ids],
        "top_ids": top_ids,
        "samples_ns": samples_ns,
    }


def run_passes(search: Callable, queries: list, passes: int) -> None:
    """Run untimed passes over every query (warm-up)."""
    for _ in range(passes):
        for query in queries:
            search(query)


def time_repetitions(search: Callable, queries: list, repeats: int) -> list[list[int]]:
    """Time every query in every repetition; returns nanoseconds [repeat][query]."""
    samples_ns: list[list[int]] = []
    for _ in range(repeats):
        repetition: list[int] = []
        for query in queries:
            started = time.perf_counter_ns()
            search(query)
            repetition.append(time.perf_counter_ns() - started)
        samples_ns.append(repetition)
    return samples_ns


def summarise_latency(samples_ns: list[list[int]]) -> dict[str, float]:
    """Mean, median, p95, min and max per-query latency in milliseconds."""
    values_ms = np.asarray(samples_ns, dtype=np.float64).ravel() / 1e6
    return {
        "mean": float(values_ms.mean()),
        "median": float(np.median(values_ms)),
        "p95": float(np.percentile(values_ms, 95)),
        "min": float(values_ms.min()),
        "max": float(values_ms.max()),
        "samples": int(values_ms.size),
    }


def agreement_at(candidate: list[int], reference: list[int], k: int) -> float | None:
    """Share of the reference top-k that the candidate's top-k also contains."""
    expected = reference[:k]
    if not expected:
        return None
    return len(set(candidate[:k]) & set(expected)) / len(expected)


def tie_aware_agreement_at(
    found: list[int], expected: list[int], found_scores: dict[int, float],
    expected_scores: dict[int, float], k: int,
) -> float | None:
    """Like agreement_at, but a result tied with the baseline's k-th one also counts."""
    expected = expected[:k]
    if not expected:
        return None
    threshold = min(expected_scores[chunk_id] for chunk_id in expected) - TIE_TOLERANCE
    hits = sum(1 for chunk_id in found[:k] if found_scores.get(chunk_id, -math.inf) >= threshold)
    return min(hits, len(expected)) / len(expected)


def summarise_agreement(per_query: list[float | None]) -> dict:
    """Mean and minimum over the queries where agreement is defined."""
    defined = [value for value in per_query if value is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "min": float(np.min(defined)) if defined else None,
        "per_query": per_query,
    }


def compare_with_reference(top_ids: list[list[int]], reference: Reference) -> dict[str, dict]:
    """Plain and tie-aware agreement@40 and @6 with the baseline."""
    found_scores = [reference.scorer(index, ids[:TOP_K]) for index, ids in enumerate(top_ids)]
    name = reference.agreement_name
    comparison: dict[str, dict] = {}
    for k in (TOP_K, SMALL_K):
        plain = [
            agreement_at(found, expected, k)
            for found, expected in zip(top_ids, reference.top_ids)
        ]
        tie_aware = [
            tie_aware_agreement_at(found, expected, scores, expected_scores, k)
            for found, expected, scores, expected_scores
            in zip(top_ids, reference.top_ids, found_scores, reference.top_scores)
        ]
        comparison[f"{name}_at_{k}"] = summarise_agreement(plain)
        comparison[f"tie_aware_{name}_at_{k}"] = summarise_agreement(tie_aware)
    return comparison


def failure_entry(name: str, stage: str, error: Exception) -> dict[str, str]:
    """Describe a method that could not be built or searched."""
    reason = f"{stage} failed: {type(error).__name__}: {error}"
    print(f"  ! {name}: {reason}", flush=True)
    return {"method": name, "reason": reason}


def unavailable_entry(methods: str, library: str, unavailable: dict[str, str]) -> dict[str, str]:
    """Describe methods skipped because their library could not be imported."""
    reason = f"{library} not importable: {unavailable.get(library, 'unknown reason')}"
    return {"method": methods, "reason": reason}


def print_progress_line(record: dict, agreement_name: str) -> None:
    """Print a one-line result as soon as a method has been measured."""
    plain = record[f"{agreement_name}_at_{TOP_K}"]["mean"]
    tie_aware = record[f"tie_aware_{agreement_name}_at_{TOP_K}"]["mean"]
    print(
        f"  {record['method']}: mean {record['latency_ms']['mean']:.3f} ms, "
        f"{agreement_name}@{TOP_K} {format_number(plain, 3)} "
        f"(tie-aware {format_number(tie_aware, 3)})",
        flush=True,
    )


# --- Index files --------------------------------------------------------------


def fresh_store_path(name: str) -> Path:
    """Return a path under data/eval/stores, removing what a previous run left there."""
    stores = STORES_DIRECTORY.resolve()
    path = (stores / name).resolve()
    if stores not in path.parents:
        raise ValueError(f"refusing to write outside {stores}: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return path


def path_size(path: Path) -> int:
    """Size in bytes of a file, or of every file below a directory."""
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


# --- Vector search ------------------------------------------------------------


def run_vector_benchmark(
    queries: list[dict],
    arguments: argparse.Namespace,
    modules: dict[str, Any],
    unavailable: dict[str, str],
    notices: list[str],
) -> dict:
    """Measure every vector-search method against the exact float32 baseline."""
    print("\n== Vector search ==", flush=True)
    corpus = load_vector_corpus()
    query_vectors, query_info = load_query_vectors(arguments.query_vectors, len(queries), corpus)
    if query_info["source"] == "stand-in":
        report_notice(notices, (
            f"NOTICE: no --query-vectors given. {len(queries)} random stored passage vectors "
            f"(seed {STAND_IN_SEED}) STAND IN for the questions. Each stand-in is its own "
            "nearest neighbour, so recall figures are only indicative."
        ))
    queries_as_rows = list(query_vectors)
    reference = make_reference(
        "recall",
        [numpy_exact_search(corpus, query) for query in queries_as_rows],
        vector_scorer(corpus, query_vectors),
    )
    builders, skipped = vector_builders(corpus, modules, unavailable)
    records, failures = measure_builders(builders, queries_as_rows, reference, arguments)
    return {
        "corpus": {
            "vector_count": int(corpus.vectors.shape[0]),
            "dimensions": int(corpus.vectors.shape[1]),
            "archive_bytes": path_size(VECTOR_ARCHIVE_PATH),
            "load_seconds": corpus.load_seconds,
        },
        "query_vectors": query_info,
        "reference_top_ids": reference.top_ids,
        "methods": records,
        "skipped": skipped + failures,
    }


def load_vector_corpus() -> VectorCorpus:
    """Load vectors.npz exactly as VectorIndex.load does, timing it."""
    started = time.perf_counter()
    with np.load(VECTOR_ARCHIVE_PATH) as archive:
        chunk_ids = archive["chunk_ids"].astype(np.int64)
        vectors = archive["vectors"].astype(np.float32)
    return VectorCorpus(chunk_ids, vectors, time.perf_counter() - started)


def load_query_vectors(
    path: Path | None, query_count: int, corpus: VectorCorpus
) -> tuple[np.ndarray, dict]:
    """Load question embeddings, or pick stored passage vectors as stand-ins."""
    if path is None:
        generator = np.random.default_rng(STAND_IN_SEED)
        positions = generator.choice(corpus.vectors.shape[0], size=query_count, replace=False)
        info = {
            "source": "stand-in",
            "seed": STAND_IN_SEED,
            "stand_in_chunk_ids": corpus.chunk_ids[positions].tolist(),
        }
        return corpus.vectors[positions].copy(), info

    loaded = np.load(path).astype(np.float32)
    expected_shape = (query_count, corpus.vectors.shape[1])
    if loaded.shape != expected_shape:
        raise SystemExit(f"--query-vectors: expected shape {expected_shape}, got {loaded.shape}")
    norms = np.linalg.norm(loaded, axis=1)
    renormalised = bool(np.any(np.abs(norms - 1.0) > 1e-3))
    if renormalised:
        print("NOTE: query vectors were not unit length; they have been L2-normalised.")
        loaded = loaded / norms[:, np.newaxis]
    info = {"source": "file", "path": str(path.resolve()), "renormalised": renormalised}
    return np.ascontiguousarray(loaded), info


def vector_builders(
    corpus: VectorCorpus, modules: dict[str, Any], unavailable: dict[str, str]
) -> tuple[list[Builder], list[dict]]:
    """List the vector methods to run, and those skipped for a missing library."""
    builders: list[Builder] = [
        ("numpy float32 exact (baseline)", partial(build_numpy_float32, corpus)),
        ("numpy float16 exact", partial(build_numpy_float16, corpus)),
    ]
    skipped: list[dict] = []
    faiss = modules.get("faiss")
    if faiss is None:
        skipped.append(unavailable_entry("faiss methods", "faiss", unavailable))
    else:
        builders.append(("faiss IndexFlatIP", partial(build_faiss_flat, faiss, corpus)))
        builders.append(("faiss HNSW", partial(build_faiss_hnsw, faiss, corpus)))
    usearch_index = modules.get("usearch")
    if usearch_index is None:
        skipped.append(unavailable_entry("usearch HNSW", "usearch", unavailable))
    else:
        builders.append(("usearch HNSW", partial(build_usearch_hnsw, usearch_index, corpus)))
    if faiss is not None:
        builders.append(("faiss SQ8 + rescore", partial(build_faiss_sq8_rescored, faiss, corpus)))
        builders.append(("faiss binary + rescore", partial(build_faiss_binary_rescored, faiss, corpus)))
    return builders, skipped


def vector_scorer(corpus: VectorCorpus, query_vectors: np.ndarray) -> Scorer:
    """Exact float32 similarity of given chunks to one query (for tie handling)."""
    position_of = {int(chunk_id): position for position, chunk_id in enumerate(corpus.chunk_ids)}

    def score(query_index: int, chunk_ids: list[int]) -> dict[int, float]:
        if not chunk_ids:
            return {}
        positions = [position_of[chunk_id] for chunk_id in chunk_ids]
        similarities = corpus.vectors[positions] @ query_vectors[query_index]
        return dict(zip(chunk_ids, similarities.tolist()))

    return score


def top_positions(similarities: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k largest similarities, best first (as VectorIndex.search)."""
    result_count = min(k, similarities.shape[0])
    top = np.argpartition(-similarities, result_count - 1)[:result_count]
    return top[np.argsort(-similarities[top])]


def ids_from_positions(chunk_ids: np.ndarray, positions: np.ndarray) -> list[int]:
    """Map index positions to chunk ids, dropping the -1 padding FAISS may return."""
    return chunk_ids[positions[positions >= 0]].tolist()


def numpy_exact_search(corpus: VectorCorpus, query: np.ndarray) -> list[int]:
    """The app's search: one matrix-vector product, then argpartition."""
    similarities = corpus.vectors @ query.astype(np.float32)
    return corpus.chunk_ids[top_positions(similarities, TOP_K)].tolist()


def build_numpy_float32(corpus: VectorCorpus) -> list[SearchMethod]:
    """The baseline: exact float32 search, as VectorIndex.search does it."""
    return [SearchMethod(
        name="numpy float32 exact (baseline)",
        search=partial(numpy_exact_search, corpus),
        build_seconds=corpus.load_seconds,
        index_bytes=int(corpus.vectors.nbytes + corpus.chunk_ids.nbytes),
        size_note="in-memory arrays; build = loading vectors.npz as VectorIndex.load does",
        settings={"library": "numpy", "dtype": "float32"},
    )]


def build_numpy_float16(corpus: VectorCorpus) -> list[SearchMethod]:
    """Exact search over a half-precision copy of the matrix."""
    started = time.perf_counter()
    vectors_half = corpus.vectors.astype(np.float16)
    build_seconds = time.perf_counter() - started

    def search(query: np.ndarray) -> list[int]:
        similarities = vectors_half @ query.astype(np.float16)
        return corpus.chunk_ids[top_positions(similarities, TOP_K)].tolist()

    return [SearchMethod(
        name="numpy float16 exact",
        search=search,
        build_seconds=build_seconds,
        index_bytes=int(vectors_half.nbytes + corpus.chunk_ids.nbytes),
        size_note="in-memory arrays; build = float32 to float16 conversion",
        settings={"library": "numpy", "dtype": "float16"},
    )]


def save_faiss_index(faiss: Any, index: Any, file_name: str, binary: bool = False) -> tuple[float, int]:
    """Write a FAISS index under data/eval/stores; return (seconds, bytes)."""
    path = fresh_store_path(file_name)
    started = time.perf_counter()
    if binary:
        faiss.write_index_binary(index, str(path))
    else:
        faiss.write_index(index, str(path))
    return time.perf_counter() - started, path_size(path)


def build_faiss_flat(faiss: Any, corpus: VectorCorpus) -> list[SearchMethod]:
    """FAISS exhaustive inner-product search."""
    started = time.perf_counter()
    index = faiss.IndexFlatIP(corpus.vectors.shape[1])
    index.add(corpus.vectors)
    build_seconds = time.perf_counter() - started
    save_seconds, index_bytes = save_faiss_index(faiss, index, "faiss_flat_ip.index")

    def search(query: np.ndarray) -> list[int]:
        _, positions = index.search(query.reshape(1, -1), TOP_K)
        return ids_from_positions(corpus.chunk_ids, positions[0])

    return [SearchMethod(
        name="faiss IndexFlatIP exact",
        search=search,
        build_seconds=build_seconds,
        save_seconds=save_seconds,
        index_bytes=index_bytes,
        size_note="index file on disk",
        settings={"library": "faiss", "index": "IndexFlatIP"},
    )]


def build_faiss_hnsw(faiss: Any, corpus: VectorCorpus) -> list[SearchMethod]:
    """One FAISS HNSW graph, searched with several efSearch values."""
    started = time.perf_counter()
    index = faiss.IndexHNSWFlat(corpus.vectors.shape[1], HNSW_CONNECTIVITY, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_CONSTRUCTION_EXPANSION
    index.add(corpus.vectors)
    build_seconds = time.perf_counter() - started
    save_seconds, index_bytes = save_faiss_index(faiss, index, f"faiss_hnsw_m{HNSW_CONNECTIVITY}.index")
    shared = {
        "build_seconds": build_seconds,
        "save_seconds": save_seconds,
        "index_bytes": index_bytes,
        "size_note": "index file on disk (vectors + graph); one build shared by all ef values",
    }
    return [faiss_hnsw_method(faiss, index, corpus, ef_search, shared) for ef_search in HNSW_SEARCH_EXPANSIONS]


def faiss_hnsw_method(
    faiss: Any, index: Any, corpus: VectorCorpus, ef_search: int, shared: dict
) -> SearchMethod:
    """A search method over a FAISS HNSW graph with a fixed efSearch."""
    parameters = faiss.SearchParametersHNSW(efSearch=ef_search)

    def search(query: np.ndarray) -> list[int]:
        _, positions = index.search(query.reshape(1, -1), TOP_K, params=parameters)
        return ids_from_positions(corpus.chunk_ids, positions[0])

    settings = {
        "library": "faiss", "index": "IndexHNSWFlat", "M": HNSW_CONNECTIVITY,
        "ef_construction": HNSW_CONSTRUCTION_EXPANSION, "ef_search": ef_search,
    }
    return SearchMethod(
        name=f"faiss HNSW M={HNSW_CONNECTIVITY} ef={ef_search}",
        search=search, settings=settings, **shared,
    )


def build_usearch_hnsw(usearch_index: Any, corpus: VectorCorpus) -> list[SearchMethod]:
    """One usearch HNSW graph, searched with several expansion_search values."""
    count, dimensions = corpus.vectors.shape
    started = time.perf_counter()
    index = usearch_index.Index(
        ndim=dimensions, metric="ip", dtype="f32",
        connectivity=HNSW_CONNECTIVITY, expansion_add=HNSW_CONSTRUCTION_EXPANSION,
    )
    index.add(np.arange(count, dtype=np.uint64), corpus.vectors)
    build_seconds = time.perf_counter() - started

    path = fresh_store_path(f"usearch_hnsw_m{HNSW_CONNECTIVITY}.usearch")
    started = time.perf_counter()
    index.save(str(path))
    shared = {
        "build_seconds": build_seconds,
        "save_seconds": time.perf_counter() - started,
        "index_bytes": path_size(path),
        "size_note": "index file on disk (vectors + graph); one build shared by all ef values",
    }
    base_settings = {
        "library": "usearch", "M": HNSW_CONNECTIVITY,
        "ef_construction": HNSW_CONSTRUCTION_EXPANSION,
        "hardware_acceleration": str(index.hardware_acceleration),
        "memory_usage_bytes": int(index.memory_usage),
    }
    return [
        usearch_method(index, corpus, ef_search, shared, base_settings)
        for ef_search in HNSW_SEARCH_EXPANSIONS
    ]


def usearch_method(
    index: Any, corpus: VectorCorpus, ef_search: int, shared: dict, base_settings: dict
) -> SearchMethod:
    """A search method over a usearch graph with a fixed expansion_search."""

    def prepare() -> None:
        index.expansion_search = ef_search

    def search(query: np.ndarray) -> list[int]:
        matches = index.search(query, TOP_K)
        return corpus.chunk_ids[matches.keys.astype(np.intp)].tolist()

    return SearchMethod(
        name=f"usearch HNSW M={HNSW_CONNECTIVITY} ef={ef_search}",
        search=search, prepare=prepare,
        settings={**base_settings, "ef_search": ef_search}, **shared,
    )


def rescore_candidates(corpus: VectorCorpus, candidates: np.ndarray, query: np.ndarray) -> list[int]:
    """Re-rank candidate positions by their exact float32 similarity."""
    candidates = candidates[candidates >= 0]
    similarities = corpus.vectors[candidates] @ query
    best = top_positions(similarities, TOP_K)
    return corpus.chunk_ids[candidates[best]].tolist()


def rescored_method(
    name: str, find_candidates: Callable, corpus: VectorCorpus, depth: int,
    shared: dict, settings: dict,
) -> SearchMethod:
    """A method that shortlists `depth` candidates, then rescores them in float32."""

    def search(query: np.ndarray) -> list[int]:
        return rescore_candidates(corpus, find_candidates(query, depth), query)

    return SearchMethod(name=name, search=search, settings={**settings, "rescore_depth": depth}, **shared)


def sq8_candidates(index: Any, query: np.ndarray, depth: int) -> np.ndarray:
    """Shortlist from an 8-bit scalar-quantised FAISS index."""
    _, positions = index.search(query.reshape(1, -1), depth)
    return positions[0]


def binary_candidates(index: Any, query: np.ndarray, depth: int) -> np.ndarray:
    """Shortlist by Hamming distance between sign bits."""
    query_bits = np.packbits(query > 0).reshape(1, -1)
    _, positions = index.search(query_bits, depth)
    return positions[0]


def build_faiss_sq8_rescored(faiss: Any, corpus: VectorCorpus) -> list[SearchMethod]:
    """8-bit scalar quantisation (per-dimension range) with float32 rescoring."""
    started = time.perf_counter()
    index = faiss.IndexScalarQuantizer(
        corpus.vectors.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    index.train(corpus.vectors)
    index.add(corpus.vectors)
    build_seconds = time.perf_counter() - started
    save_seconds, index_bytes = save_faiss_index(faiss, index, "faiss_sq8.index")
    shared = {
        "build_seconds": build_seconds, "save_seconds": save_seconds, "index_bytes": index_bytes,
        "size_note": "int8 codes on disk; rescoring also reads the float32 matrix",
    }
    settings = {"library": "faiss", "index": "IndexScalarQuantizer QT_8bit",
                "rescore_vectors_bytes": int(corpus.vectors.nbytes)}
    return [
        rescored_method(f"faiss SQ8 + f32 rescore {depth}", partial(sq8_candidates, index),
                        corpus, depth, shared, settings)
        for depth in RESCORE_DEPTHS
    ]


def build_faiss_binary_rescored(faiss: Any, corpus: VectorCorpus) -> list[SearchMethod]:
    """1-bit sign quantisation (Hamming search) with float32 rescoring."""
    started = time.perf_counter()
    bits = np.packbits(corpus.vectors > 0, axis=1)
    index = faiss.IndexBinaryFlat(corpus.vectors.shape[1])
    index.add(bits)
    build_seconds = time.perf_counter() - started
    save_seconds, index_bytes = save_faiss_index(faiss, index, "faiss_binary.index", binary=True)
    shared = {
        "build_seconds": build_seconds, "save_seconds": save_seconds, "index_bytes": index_bytes,
        "size_note": "1-bit codes on disk; rescoring also reads the float32 matrix",
    }
    settings = {"library": "faiss", "index": "IndexBinaryFlat (sign bits)",
                "rescore_vectors_bytes": int(corpus.vectors.nbytes)}
    return [
        rescored_method(f"faiss binary + f32 rescore {depth}", partial(binary_candidates, index),
                        corpus, depth, shared, settings)
        for depth in RESCORE_DEPTHS
    ]


# --- Keyword search -----------------------------------------------------------


def run_keyword_benchmark(
    queries: list[dict],
    arguments: argparse.Namespace,
    modules: dict[str, Any],
    unavailable: dict[str, str],
) -> dict:
    """Measure every keyword-search method against the app's FTS5 query."""
    print("\n== Keyword search ==", flush=True)
    connection = open_index_read_only()
    try:
        chunk_rows = load_chunk_rows(connection)
        builder = KeywordQueryBuilder()
        questions = [query["text"] for query in queries]
        reference = make_reference(
            "overlap",
            [search_fts5_app(connection, builder, question) for question in questions],
            fts5_scorer(connection, [builder.build(question) for question in questions]),
        )
        builders, skipped = keyword_builders(connection, chunk_rows, modules, unavailable)
        records, failures = measure_builders(builders, questions, reference, arguments)
        corpus = {
            "chunk_rows": len(chunk_rows),
            "fts_rows": connection.execute("SELECT COUNT(*) FROM chunk_search").fetchone()[0],
            "characters": sum(len(text) for _, text in chunk_rows),
        }
    finally:
        connection.close()
    return {
        "corpus": corpus,
        "queries": describe_keyword_queries(queries, builder),
        "reference_top_ids": reference.top_ids,
        "methods": records,
        "skipped": skipped + failures,
    }


def open_index_read_only() -> sqlite3.Connection:
    """Open data/index.db read-only, with the app's row factory and temp store."""
    uri = DATABASE_PATH.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def load_chunk_rows(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    """Every (chunk_id, text) pair, in chunk-id order."""
    rows = connection.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    return [(int(row[0]), row[1]) for row in rows]


def describe_keyword_queries(queries: list[dict], builder: KeywordQueryBuilder) -> list[dict]:
    """Each question's id and how many search terms the engines receive.

    The question text and its terms are left out: the repository is public and
    some questions name staff (see evaluation/privacy.py).
    """
    described = []
    for query in queries:
        expression = builder.build(query["text"])
        described.append({
            "id": query.get("id"),
            "term_count": len(expression_tokens(expression)),
        })
    return described


def keyword_builders(
    connection: sqlite3.Connection,
    chunk_rows: list[tuple[int, str]],
    modules: dict[str, Any],
    unavailable: dict[str, str],
) -> tuple[list[Builder], list[dict]]:
    """List the keyword methods to run, and those skipped for a missing library."""
    builders: list[Builder] = [("fts5", partial(build_fts5_methods, connection, chunk_rows))]
    skipped: list[dict] = []
    bm25s = modules.get("bm25s")
    if bm25s is None:
        skipped.append(unavailable_entry("bm25s", "bm25s", unavailable))
    else:
        builders.append(("bm25s", partial(build_bm25s_methods, bm25s, chunk_rows)))
    tantivy = modules.get("tantivy")
    if tantivy is None:
        skipped.append(unavailable_entry("tantivy", "tantivy", unavailable))
    else:
        builders.append(("tantivy", partial(build_tantivy_methods, tantivy, chunk_rows)))
    return builders, skipped


def fold_case_and_diacritics(text: str) -> str:
    """Lowercase and strip diacritics, as unicode61 remove_diacritics 2 does."""
    lowered = text.lower()
    if lowered.isascii():
        return lowered
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(character for character in decomposed if not unicodedata.combining(character))
    return unicodedata.normalize("NFC", stripped)


def fts5_like_tokens(text: str) -> list[str]:
    """Tokenise text the way FTS5's unicode61 tokenizer does (approximately)."""
    return FTS5_TOKEN_PATTERN.findall(fold_case_and_diacritics(text))


def expression_tokens(expression: str) -> list[str]:
    """The distinct tokens FTS5 would look up for a KeywordQueryBuilder expression."""
    tokens: list[str] = []
    for term in QUOTED_TERM_PATTERN.findall(expression):
        tokens.extend(fts5_like_tokens(term))
    return list(dict.fromkeys(tokens))


# FTS5 --------------------------------------------------------------------------


def search_fts5_app(connection: sqlite3.Connection, builder: KeywordQueryBuilder, question: str) -> list[int]:
    """The app's keyword search: builder, then the joined BM25 query."""
    expression = builder.build(question)
    if not expression:
        return []
    rows = connection.execute(APP_KEYWORD_SQL, (expression, TOP_K)).fetchall()
    return [int(row["chunk_id"]) for row in rows]


def search_fts5_rowids(connection: sqlite3.Connection, builder: KeywordQueryBuilder, question: str) -> list[int]:
    """The same BM25 ranking, returning rowids only (no joins, no text)."""
    expression = builder.build(question)
    if not expression:
        return []
    rows = connection.execute(ROWID_ONLY_SQL, (expression, TOP_K)).fetchall()
    return [int(row[0]) for row in rows]


def search_fts5_rank_first(connection: sqlite3.Connection, builder: KeywordQueryBuilder, question: str) -> list[int]:
    """The app's columns, ranking first and joining only the top rows."""
    expression = builder.build(question)
    if not expression:
        return []
    rows = connection.execute(RANK_FIRST_SQL, (expression, TOP_K)).fetchall()
    return [int(row["chunk_id"]) for row in rows]


def fts5_scorer(connection: sqlite3.Connection, expressions: list[str]) -> Scorer:
    """FTS5 relevance (-bm25, higher is better) of given chunks for one query."""

    def score(query_index: int, chunk_ids: list[int]) -> dict[int, float]:
        expression = expressions[query_index]
        if not chunk_ids or not expression:
            return {}
        sql = FTS5_SCORE_SQL.format(placeholders=",".join("?" for _ in chunk_ids))
        rows = connection.execute(sql, [expression, *chunk_ids]).fetchall()
        return {int(row[0]): -float(row[1]) for row in rows}

    return score


def build_fts5_copy(chunk_rows: list[tuple[int, str]]) -> tuple[float, int]:
    """Time filling a fresh copy of the app's FTS5 table; return (seconds, bytes)."""
    path = fresh_store_path("fts5_copy.db")
    fresh_store_path("fts5_copy.db-journal")
    started = time.perf_counter()
    connection = sqlite3.connect(path)
    try:
        connection.execute(FTS5_CREATE_SQL)
        connection.executemany("INSERT INTO chunk_search(rowid, text) VALUES (?, ?)", chunk_rows)
        connection.execute("INSERT INTO chunk_search(chunk_search) VALUES('optimize')")
        connection.commit()
        seconds = time.perf_counter() - started
    finally:
        connection.close()
    return seconds, path_size(path)


def build_fts5_methods(connection: sqlite3.Connection, chunk_rows: list[tuple[int, str]]) -> list[SearchMethod]:
    """The baseline (app SQL) and a join-free variant, both on the real index.db."""
    build_seconds, index_bytes = build_fts5_copy(chunk_rows)
    builder = KeywordQueryBuilder()
    shared = {
        "build_seconds": build_seconds,
        "index_bytes": index_bytes,
        "size_note": "a fresh SQLite file holding only the FTS5 table (which keeps its own "
                     "copy of the text); build is one bulk transaction + optimize, on disk",
    }
    settings = {"library": "sqlite fts5", "sqlite": sqlite3.sqlite_version,
                "tokenizer": "unicode61 remove_diacritics 2", "k1": 1.2, "b": 0.75}
    return [
        SearchMethod(name="fts5 app SQL with joins (baseline)",
                     search=partial(search_fts5_app, connection, builder),
                     settings={**settings, "sql": "app"}, **shared),
        SearchMethod(name="fts5 rowids only, no joins",
                     search=partial(search_fts5_rowids, connection, builder),
                     settings={**settings, "sql": ROWID_ONLY_SQL}, **shared),
        SearchMethod(name=f"fts5 rank first, join top {TOP_K}",
                     search=partial(search_fts5_rank_first, connection, builder),
                     settings={**settings, "sql": RANK_FIRST_SQL}, **shared),
    ]


# bm25s -------------------------------------------------------------------------


def tokenise_corpus(chunk_rows: list[tuple[int, str]]) -> TokenisedCorpus:
    """Tokenise every chunk FTS5-style and map tokens to integer ids."""
    started = time.perf_counter()
    vocabulary: dict[str, int] = {}
    token_ids: list[list[int]] = []
    for _, text in chunk_rows:
        token_ids.append([vocabulary.setdefault(token, len(vocabulary)) for token in fts5_like_tokens(text)])
    chunk_ids = np.array([chunk_id for chunk_id, _ in chunk_rows], dtype=np.int64)
    return TokenisedCorpus(chunk_ids, token_ids, vocabulary, time.perf_counter() - started)


def build_bm25s_methods(bm25s: Any, chunk_rows: list[tuple[int, str]]) -> list[SearchMethod]:
    """Tokenise once, then index each bm25s scoring variant."""
    corpus = tokenise_corpus(chunk_rows)
    return [build_one_bm25s(bm25s, corpus, label, parameters) for label, parameters in BM25S_VARIANTS]


def build_one_bm25s(bm25s: Any, corpus: TokenisedCorpus, label: str, parameters: dict) -> SearchMethod:
    """Index one bm25s variant and save it under data/eval/stores."""
    retriever = bm25s.BM25(**parameters)
    started = time.perf_counter()
    # bm25s adds an empty token to the vocabulary it is given, so pass a copy.
    retriever.index((corpus.token_ids, dict(corpus.vocabulary)), show_progress=False)
    index_seconds = time.perf_counter() - started

    path = fresh_store_path(f"bm25s_{parameters['method']}")
    started = time.perf_counter()
    retriever.save(str(path))
    save_seconds = time.perf_counter() - started

    return SearchMethod(
        name=f"bm25s {parameters['method']} k1={parameters['k1']} ({label})",
        search=partial(search_bm25s, retriever, corpus.chunk_ids, KeywordQueryBuilder()),
        build_seconds=corpus.seconds + index_seconds,
        save_seconds=save_seconds,
        index_bytes=path_size(path),
        size_note="saved index directory (sparse score matrix + vocabulary); "
                  "build = FTS5-like tokenisation + indexing",
        settings={
            "library": "bm25s", **parameters,
            "tokenise_seconds": corpus.seconds,
            "index_seconds": index_seconds,
            "token_count": sum(len(ids) for ids in corpus.token_ids),
            "vocabulary_size": len(corpus.vocabulary),
        },
    )


def search_bm25s(retriever: Any, chunk_ids: np.ndarray, builder: KeywordQueryBuilder, question: str) -> list[int]:
    """Query bm25s with the tokens of the app's FTS5 expression."""
    tokens = expression_tokens(builder.build(question))
    if not tokens:
        return []
    documents, scores = retriever.retrieve([tokens], k=TOP_K, show_progress=False)
    # bm25s pads with zero-score documents; FTS5 returns matching rows only.
    matched = documents[0][scores[0] > 0]
    return chunk_ids[matched].tolist()


# tantivy -----------------------------------------------------------------------


def create_tantivy_index(tantivy: Any, path: Path) -> Any:
    """An on-disk tantivy index whose text analyzer imitates unicode61."""
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("chunk_id", stored=True)
    schema_builder.add_text_field("text", stored=False, tokenizer_name=TANTIVY_TOKENIZER)
    index = tantivy.Index(schema_builder.build(), path=str(path), reuse=False)
    analyzer = (
        tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
        .filter(tantivy.Filter.lowercase())
        .filter(tantivy.Filter.ascii_fold())
        .build()
    )
    index.register_tokenizer(TANTIVY_TOKENIZER, analyzer)
    return index


def build_tantivy_methods(tantivy: Any, chunk_rows: list[tuple[int, str]]) -> list[SearchMethod]:
    """Index every chunk in tantivy (written to disk as it builds)."""
    path = fresh_store_path("tantivy")
    path.mkdir(parents=True)
    started = time.perf_counter()
    index = create_tantivy_index(tantivy, path)
    writer = index.writer()
    for chunk_id, text in chunk_rows:
        document = tantivy.Document()
        document.add_integer("chunk_id", chunk_id)
        document.add_text("text", text)
        writer.add_document(document)
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    build_seconds = time.perf_counter() - started
    searcher = index.searcher()

    return [SearchMethod(
        name="tantivy BM25 (k1=1.2, Lucene idf)",
        search=partial(search_tantivy, index, searcher, KeywordQueryBuilder()),
        build_seconds=build_seconds,
        index_bytes=path_size(path),
        size_note="index directory on disk (positions included, text not stored)",
        settings={
            "library": "tantivy",
            "analyzer": "simple tokenizer + lowercase + ascii_fold",
            "segments": searcher.num_segments,
            "query": "the app's FTS5 expression, parsed by tantivy's query parser",
        },
    )]


def search_tantivy(index: Any, searcher: Any, builder: KeywordQueryBuilder, question: str) -> list[int]:
    """Parse the app's FTS5 expression (valid tantivy syntax too) and search."""
    expression = builder.build(question)
    if not expression:
        return []
    query = index.parse_query(expression, ["text"])
    hits = searcher.search(query, limit=TOP_K, count=False).hits
    return [searcher.doc(address).get_first("chunk_id") for _, address in hits]


# --- Machine description ------------------------------------------------------

WINDOWS_HARDWARE_QUERY = (
    "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; "
    "$system = Get-CimInstance Win32_ComputerSystem; "
    "[pscustomobject]@{ cpu_name = $cpu.Name.Trim(); physical_cores = $cpu.NumberOfCores; "
    "logical_processors = $cpu.NumberOfLogicalProcessors; max_clock_mhz = $cpu.MaxClockSpeed; "
    "ram_bytes = $system.TotalPhysicalMemory } | ConvertTo-Json -Compress"
)


def describe_parameters(arguments: argparse.Namespace, query_count: int) -> dict:
    """The settings this run used."""
    return {
        "warmup_passes": arguments.warmup,
        "timed_repetitions": arguments.repeats,
        "only": arguments.only,
        "query_count": query_count,
        "query_vectors_argument": str(arguments.query_vectors) if arguments.query_vectors else None,
        "top_k": TOP_K,
        "small_k": SMALL_K,
        "hnsw": {
            "M": HNSW_CONNECTIVITY,
            "ef_construction": HNSW_CONSTRUCTION_EXPANSION,
            "ef_search": list(HNSW_SEARCH_EXPANSIONS),
        },
        "rescore_depths": list(RESCORE_DEPTHS),
        "stand_in_seed": STAND_IN_SEED,
        "bm25s_variants": [{"label": label, **parameters} for label, parameters in BM25S_VARIANTS],
        "stores_directory": str(STORES_DIRECTORY),
    }


def describe_machine() -> dict:
    """CPU, memory, OS, Python and clock details."""
    clock = time.get_clock_info("perf_counter")
    machine = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "sqlite": sqlite3.sqlite_version,
        "perf_counter": {"implementation": clock.implementation, "resolution_s": clock.resolution},
    }
    if sys.platform == "win32":
        machine.update(read_windows_hardware())
    return machine


def read_windows_hardware() -> dict:
    """CPU name, core counts and RAM, from Windows' CIM classes."""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_HARDWARE_QUERY],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"hardware_query_error": f"{type(error).__name__}: {error}"}


def library_versions() -> dict[str, str | None]:
    """Installed versions of the libraries this benchmark depends on."""
    versions: dict[str, str | None] = {}
    for distribution in RECORDED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def describe_runtime(faiss: Any) -> dict:
    """Thread pools in effect (left at library defaults) and the FAISS build loaded."""
    return {
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "faiss_omp_max_threads": faiss.omp_get_max_threads() if faiss is not None else None,
        "faiss_compile_options": faiss.get_compile_options() if faiss is not None else None,
        "faiss_loaded_modules": sorted(name for name in sys.modules if name.startswith("faiss.swigfaiss")),
        "numpy_build": numpy_build_info(),
        "threadpools": threadpool_snapshot(),
    }


def numpy_build_info() -> Any:
    """NumPy's BLAS/LAPACK build configuration, if NumPy can report it."""
    try:
        return np.show_config(mode="dicts").get("Build Dependencies")
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def threadpool_snapshot() -> Any:
    """Loaded BLAS/OpenMP pools and their thread counts, via threadpoolctl."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return threadpool_info()


# --- Output -------------------------------------------------------------------


def json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays and paths for json.dumps."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_report(report: dict, started_at: datetime) -> Path:
    """Write the full report to evaluation_results/stores_<timestamp>.json."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIRECTORY / f"stores_{started_at:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=json_default), encoding="utf-8")
    return path


def format_number(value: float | None, decimals: int) -> str:
    """A fixed-precision number, or n/a."""
    return "n/a" if value is None else f"{value:.{decimals}f}"


def format_megabytes(byte_count: int | None) -> str:
    """Bytes as MiB with one decimal, or n/a."""
    return "n/a" if byte_count is None else f"{byte_count / 1024 ** 2:.1f}"


def markdown_table(records: list[dict], agreement_name: str) -> str:
    """One Markdown row per measured method."""
    header = (
        "| Method | Build s | Size MiB | Mean ms | Median ms | p95 ms | Min ms | Max ms "
        f"| {agreement_name}@{TOP_K} | {agreement_name}@{SMALL_K} "
        f"| tie-aware @{TOP_K} | tie-aware @{SMALL_K} |"
    )
    lines = [header, "|---" + "|---:" * 11 + "|"]
    for record in records:
        latency = record["latency_ms"]
        cells = [
            record["method"],
            format_number(record["build_seconds"], 2),
            format_megabytes(record["index_bytes"]),
            format_number(latency["mean"], 3),
            format_number(latency["median"], 3),
            format_number(latency["p95"], 3),
            format_number(latency["min"], 3),
            format_number(latency["max"], 3),
        ]
        for prefix in ("", "tie_aware_"):
            for k in (TOP_K, SMALL_K):
                cells.append(format_number(record[f"{prefix}{agreement_name}_at_{k}"]["mean"], 3))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_summary(report: dict) -> None:
    """Print a compact Markdown summary of both benchmarks."""
    parameters = report["parameters"]
    print(f"\n## Search store benchmark ({parameters['warmup_passes']} warm-up passes, "
          f"{parameters['timed_repetitions']} timed repetitions x {parameters['query_count']} queries)")
    if "vector" in report:
        source = report["vector"]["query_vectors"]["source"]
        label = "STAND-IN stored passage vectors" if source == "stand-in" else "question embeddings"
        print(f"\n### Vector search, top {TOP_K} (query vectors: {label}; recall vs numpy float32 exact)\n")
        print(markdown_table(report["vector"]["methods"], "recall"))
        print_skipped(report["vector"]["skipped"])
    if "keyword" in report:
        print(f"\n### Keyword search, top {TOP_K} (overlap vs FTS5 app SQL)\n")
        print(markdown_table(report["keyword"]["methods"], "overlap"))
        print_skipped(report["keyword"]["skipped"])
    for notice in report["notices"]:
        print(f"\n{notice}")


def print_skipped(skipped: list[dict]) -> None:
    """List methods that were skipped or failed."""
    for entry in skipped:
        print(f"- skipped {entry['method']}: {entry['reason']}")


if __name__ == "__main__":
    sys.exit(main())
