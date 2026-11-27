"""Tests for rank fusion, the corpus scanner and document value objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.documents import ExtractedSection, ExtractionOutcome, RetrievedPassage, SourceDocument
from app.extractors import ExtractorRegistry
from app.ingestion import CorpusRoot, CorpusScanner
from app.retrieval import ReciprocalRankFusion


def build_passage(chunk_id: int, dense: float | None = None, sparse: float | None = None):
    """Construct a retrieval result with the given scores."""
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_name=f"berkas{chunk_id}.pdf",
        relative_path=f"folder/berkas{chunk_id}.pdf",
        absolute_path=f"/corpus/folder/berkas{chunk_id}.pdf",
        section_label="Halaman 1",
        text=f"isi potongan {chunk_id}",
        dense_score=dense,
        sparse_score=sparse,
    )


class TestSourceDocument:
    """Identity and derived metadata of a discovered file."""

    def test_identifier_is_stable_for_the_same_path(self):
        first = SourceDocument(Path("/c/a.pdf"), "a/b.pdf", "b.pdf", ".pdf", 10, 1.0)
        second = SourceDocument(Path("/c/a.pdf"), "a/b.pdf", "b.pdf", ".pdf", 99, 2.0)
        assert first.document_id == second.document_id

    def test_identifier_differs_between_paths(self):
        first = SourceDocument(Path("/c/a.pdf"), "a/b.pdf", "b.pdf", ".pdf", 10, 1.0)
        second = SourceDocument(Path("/c/c.pdf"), "a/c.pdf", "c.pdf", ".pdf", 10, 1.0)
        assert first.document_id != second.document_id

    def test_signature_tracks_size_and_modification_time(self):
        document = SourceDocument(Path("/c/a.pdf"), "a.pdf", "a.pdf", ".pdf", 2048, 1700.9)
        assert document.content_signature == "2048:1700"

    def test_signature_carries_the_extractor_revision_of_its_format(self):
        """A new Word reader must re-read every .docx once, and nothing else."""
        word = SourceDocument(Path("/c/a.docx"), "a.docx", "a.docx", ".docx", 2048, 1700.9)
        assert word.content_signature == "2048:1700:r3"

    def test_top_level_folder_is_the_first_path_segment(self):
        document = SourceDocument(
            Path("/c/x.pdf"), "1. Budaya Mutu/sub/x.pdf", "x.pdf", ".pdf", 1, 1.0
        )
        assert document.top_level_folder == "1. Budaya Mutu"

    def test_root_level_files_report_a_root_folder(self):
        document = SourceDocument(Path("/c/x.pdf"), "x.pdf", "x.pdf", ".pdf", 1, 1.0)
        assert document.top_level_folder == "(root)"


class TestExtractionOutcome:
    """Success reporting and character accounting."""

    def test_outcome_with_sections_succeeds(self):
        outcome = ExtractionOutcome(sections=[ExtractedSection(1, "Halaman 1", "isi")])
        assert outcome.succeeded is True

    def test_outcome_without_sections_fails(self):
        assert ExtractionOutcome().succeeded is False

    def test_outcome_with_an_error_fails_even_with_sections(self):
        outcome = ExtractionOutcome(
            sections=[ExtractedSection(1, "Halaman 1", "isi")], error_message="rusak"
        )
        assert outcome.succeeded is False

    def test_total_characters_sums_every_section(self):
        outcome = ExtractionOutcome(
            sections=[
                ExtractedSection(1, "Halaman 1", "12345"),
                ExtractedSection(2, "Halaman 2", "123"),
            ]
        )
        assert outcome.total_characters == 8

    def test_failure_factory_records_the_exception_type(self):
        outcome = ExtractionOutcome.from_failure(ValueError("berkas rusak"))
        assert "ValueError" in outcome.error_message
        assert "berkas rusak" in outcome.error_message


class TestRetrievedPassage:
    """How a passage reports the way it was found."""

    def test_dense_only_passage_reports_semantic(self):
        assert build_passage(1, dense=0.8).retrieval_methods == ["semantic"]

    def test_sparse_only_passage_reports_keyword(self):
        assert build_passage(1, sparse=3.1).retrieval_methods == ["keyword"]

    def test_passage_found_by_both_reports_both(self):
        assert build_passage(1, dense=0.8, sparse=3.1).retrieval_methods == [
            "semantic",
            "keyword",
        ]

    def test_ranking_score_prefers_the_reranker(self):
        passage = build_passage(1, dense=0.8)
        passage.fused_score = 0.02
        passage.reranked_score = 7.5
        assert passage.ranking_score == 7.5

    def test_ranking_score_falls_back_to_fusion(self):
        passage = build_passage(1, dense=0.8)
        passage.fused_score = 0.02
        assert passage.ranking_score == pytest.approx(0.02)


class TestReciprocalRankFusion:
    """Merging the dense and sparse ranked lists."""

    def test_a_passage_in_both_lists_outranks_one_in_a_single_list(self):
        shared_dense = build_passage(1, dense=0.5)
        shared_sparse = build_passage(1, sparse=2.0)
        dense_only = build_passage(2, dense=0.9)

        fused = ReciprocalRankFusion(60).fuse([[dense_only, shared_dense], [shared_sparse]])
        assert fused[0].chunk_id == 1

    def test_duplicate_hits_are_merged_into_one_result(self):
        fused = ReciprocalRankFusion(60).fuse(
            [[build_passage(1, dense=0.5)], [build_passage(1, sparse=2.0)]]
        )
        assert len(fused) == 1

    def test_merging_preserves_scores_from_both_lists(self):
        fused = ReciprocalRankFusion(60).fuse(
            [[build_passage(1, dense=0.5)], [build_passage(1, sparse=2.0)]]
        )
        assert fused[0].dense_score == pytest.approx(0.5)
        assert fused[0].sparse_score == pytest.approx(2.0)

    def test_results_are_ordered_by_fused_score(self):
        fused = ReciprocalRankFusion(60).fuse(
            [[build_passage(1), build_passage(2), build_passage(3)]]
        )
        assert [passage.chunk_id for passage in fused] == [1, 2, 3]

    def test_higher_rank_contributes_more(self):
        fusion = ReciprocalRankFusion(60)
        top_ranked = fusion.fuse([[build_passage(1), build_passage(2)]])
        assert top_ranked[0].fused_score > top_ranked[1].fused_score

    def test_empty_input_produces_an_empty_result(self):
        assert ReciprocalRankFusion(60).fuse([[], []]) == []

    def test_smoothing_constant_damps_rank_differences(self):
        """A larger constant flattens the gap between rank 1 and rank 2."""
        sharp = ReciprocalRankFusion(1).fuse([[build_passage(1), build_passage(2)]])
        flat = ReciprocalRankFusion(1000).fuse([[build_passage(1), build_passage(2)]])
        sharp_gap = sharp[0].fused_score - sharp[1].fused_score
        flat_gap = flat[0].fused_score - flat[1].fused_score
        assert sharp_gap > flat_gap


class TestCorpusScanner:
    """Walking the corpus and applying the inclusion rules."""

    @pytest.fixture
    def scanner(self, isolated_settings, guard):
        registry = ExtractorRegistry(isolated_settings, guard)
        return CorpusScanner(isolated_settings, guard, registry)

    def test_finds_indexable_files(self, scanner, sample_corpus):
        found = {document.display_name for document in scanner.scan()}
        assert {"catatan.txt", "data.csv", "laporan.md", "config.json"} <= found

    def test_excludes_the_developer_directory(self, scanner, sample_corpus):
        """`zdev` holds scratch scripts and must never be indexed."""
        relative_paths = [document.relative_path for document in scanner.scan()]
        assert not any("zdev" in path for path in relative_paths)

    def test_skips_video_and_shortcut_files(self, scanner, sample_corpus):
        found = {document.display_name for document in scanner.scan()}
        assert "rekaman.mov" not in found
        assert "pintasan.lnk" not in found

    def test_records_relative_paths(self, scanner, sample_corpus):
        by_name = {document.display_name: document for document in scanner.scan()}
        assert by_name["catatan.txt"].relative_path == str(Path("1. Budaya Mutu/catatan.txt"))

    def test_results_are_sorted_for_predictable_progress(self, scanner, sample_corpus):
        relative_paths = [document.relative_path.lower() for document in scanner.scan()]
        assert relative_paths == sorted(relative_paths)

    def test_skips_office_lock_files(self, scanner, isolated_settings, sample_corpus):
        """Word and Excel leave `~$name.docx` lock files beside open documents."""
        (isolated_settings.source_directory / "~$laporan.docx").write_bytes(b"lock")
        assert not any(
            document.display_name.startswith("~$") for document in scanner.scan()
        )

    def test_skips_files_above_the_size_limit(self, scanner, isolated_settings, sample_corpus):
        """A single oversized file is dropped while its smaller siblings remain."""
        oversized = isolated_settings.source_directory / "arsip-besar.txt"
        oversized.write_bytes(b"x" * 300_000)
        isolated_settings.maximum_file_megabytes = 0.2  # ~210 KB

        found = {document.display_name for document in scanner.scan()}
        assert "arsip-besar.txt" not in found
        assert "laporan.md" in found

    def test_missing_source_directory_raises(self, isolated_settings, guard):
        isolated_settings.source_directory = isolated_settings.source_directory / "hilang"
        registry = ExtractorRegistry(isolated_settings, guard)
        with pytest.raises(FileNotFoundError):
            CorpusScanner(isolated_settings, guard, registry).scan()

    def test_extra_roots_are_not_scanned_by_default_in_tests(
        self, scanner, isolated_settings, sample_corpus
    ):
        """The fixture clears extra roots, so only the primary corpus is walked."""
        isolated_settings.additional_source_directories = []
        assert all(
            "corpus-lokal" not in document.relative_path for document in scanner.scan()
        )

    def test_scanning_does_not_modify_the_corpus(self, scanner, sample_corpus):
        """The scan is pure observation: no file's bytes may change."""
        before = {
            path: path.read_bytes()
            for path in sample_corpus.rglob("*")
            if path.is_file()
        }
        scanner.scan()
        after = {
            path: path.read_bytes()
            for path in sample_corpus.rglob("*")
            if path.is_file()
        }
        assert before == after


class TestCorpusRoot:
    """Labelling paths so several roots can share one index."""

    def test_primary_root_leaves_paths_unchanged(self):
        """Shared-drive paths must read exactly as they do on the drive."""
        root = CorpusRoot(Path("/corpus"), label="")
        assert root.relative_path_for(Path("/corpus/a/b.pdf")) == str(Path("a/b.pdf"))

    def test_labelled_root_prefixes_paths(self):
        root = CorpusRoot(Path("/lokal"), label="corpus-lokal")
        assert root.relative_path_for(Path("/lokal/x.xlsx")) == str(Path("corpus-lokal/x.xlsx"))

    def test_primary_root_is_required(self):
        assert CorpusRoot(Path("/corpus"), "").is_required is True


class TestMultipleCorpusRoots:
    """Indexing the shared drive alongside the project's local copy."""

    @pytest.fixture
    def scanner(self, isolated_settings, guard):
        registry = ExtractorRegistry(isolated_settings, guard)
        return CorpusScanner(isolated_settings, guard, registry)

    def test_primary_root_is_listed_first(self, scanner, isolated_settings, extra_corpus_root):
        roots = scanner.build_roots()
        assert roots[0].directory == isolated_settings.source_directory
        assert roots[0].label == ""

    def test_extra_root_is_labelled_by_its_directory_name(self, scanner, extra_corpus_root):
        assert scanner.build_roots()[1].label == "corpus-lokal"

    def test_documents_from_both_roots_are_found(
        self, scanner, sample_corpus, extra_corpus_root
    ):
        found = {document.display_name for document in scanner.scan()}
        assert "laporan.md" in found
        assert "data_dosen_sinta.xlsx.txt" in found

    def test_extra_root_documents_carry_the_label(
        self, scanner, sample_corpus, extra_corpus_root
    ):
        by_name = {document.display_name: document for document in scanner.scan()}
        mapping_document = by_name["data_dosen_sinta.xlsx.txt"]
        assert mapping_document.relative_path.startswith("corpus-lokal")
        assert mapping_document.top_level_folder == "corpus-lokal"

    def test_identifiers_stay_unique_across_roots(
        self, scanner, isolated_settings, sample_corpus, extra_corpus_root
    ):
        """The same filename under two roots must not collapse into one entry."""
        (extra_corpus_root / "laporan.md").write_text("salinan lokal", encoding="utf-8")
        documents = scanner.scan()
        identifiers = [document.document_id for document in documents]
        assert len(identifiers) == len(set(identifiers))

    def test_missing_extra_root_is_skipped_silently(
        self, scanner, isolated_settings, sample_corpus, tmp_path
    ):
        """A colleague without the local copy must still be able to index."""
        isolated_settings.additional_source_directories = [tmp_path / "tidak-ada"]
        assert scanner.scan()

    def test_missing_primary_root_still_raises(
        self, scanner, isolated_settings, extra_corpus_root
    ):
        isolated_settings.source_directory = isolated_settings.source_directory / "hilang"
        with pytest.raises(FileNotFoundError):
            scanner.scan()


class FixedQuestionEmbedder:
    """Embeds every question as the same vector, so the stored vectors decide."""

    def encode_question(self, question: str):
        import numpy as np

        return np.array([1, 0, 0, 0], dtype=np.float32)


PUBLICATIONS_TABLE = "Tabel 3.C.2 Publikasi Penelitian"


class TestNamedTables:
    """A question naming a table by number gets that table's rows first."""

    TABLE = PUBLICATIONS_TABLE
    ROWS = [
        f"{PUBLICATIONS_TABLE}\nNo | Nama DTPR | Judul Publikasi\n\n"
        f"{number} | Dosen {number} | Judul nomor {number} tentang topik {topic}"
        for number, topic in ((1, "jaringan"), (2, "basis data"), (3, "keamanan"))
    ]

    @pytest.fixture
    def retriever(self, isolated_settings, guard):
        import numpy as np

        from app.documents import TextChunk
        from app.reranking import CrossEncoderReranker
        from app.retrieval import HybridRetriever
        from app.storage import DocumentCatalogue, VectorIndex

        isolated_settings.embedding_dimensions = 4
        isolated_settings.reranking_enabled = False
        catalogue = DocumentCatalogue(isolated_settings, guard)
        vectors = VectorIndex(isolated_settings, guard)

        def store(
            path: str, chunks: list[tuple[str, str]], similarity: float, modified: float = 1.0
        ) -> None:
            document = SourceDocument(Path("/corpus") / path, path, Path(path).name, ".docx", 100, modified)
            rows = [
                TextChunk(document.document_id, index, label, text, len(text.split()))
                for index, (label, text) in enumerate(chunks)
            ]
            chunk_ids = catalogue.save_document(
                document, rows, section_count=len(rows), character_count=100,
                scanned_section_count=0, ocr_section_count=0, status="indexed",
            )
            # Lower similarity for later rows, so their order is known.
            matrix = np.array(
                [[similarity - 0.01 * index, 1 - similarity, 0, 0] for index in range(len(rows))],
                dtype=np.float32,
            )
            vectors.add(chunk_ids, matrix / np.linalg.norm(matrix, axis=1, keepdims=True))

        # The narrative mentioning the table is closest by meaning of all.
        store("LED.docx", [("Dokumen", f"Rincian publikasi dosen ada pada {self.TABLE}.")] * 1, 0.99)
        # The draft is the closer by meaning; the final is the newer file.
        store("LKPS draf.docx", [(self.TABLE, row + " (draf)") for row in self.ROWS], 0.7, modified=1.0)
        store("LKPS final.docx", [(self.TABLE, row) for row in self.ROWS], 0.6, modified=2.0)
        store("Lain.docx", [("Tabel 3.C.20 Lain", "Tabel 3.C.20 Lain\n\nbaris tabel lain sama sekali")], 0.95)
        yield HybridRetriever(isolated_settings, catalogue, vectors, FixedQuestionEmbedder(),
                              CrossEncoderReranker(isolated_settings))
        catalogue.close()

    def test_the_named_table_takes_the_first_places(self, retriever):
        passages = retriever.retrieve("Tabel 3.C.2 Publikasi Penelitian")
        assert [p.section_label for p in passages[:3]] == [self.TABLE] * 3

    def test_each_row_appears_once_across_versions_of_the_document(self, retriever):
        passages = retriever.retrieve("Tabel 3.C.2 Publikasi Penelitian")
        rows = [p.text for p in passages if p.section_label == self.TABLE]
        assert sorted(rows) == sorted(self.ROWS)

    def test_rows_come_from_the_newest_version_first(self, retriever):
        passages = retriever.retrieve("Tabel 3.C.2 Publikasi Penelitian")
        assert {p.document_name for p in passages[:3]} == {"LKPS final.docx"}

    def test_other_places_are_still_filled_by_ranking(self, retriever):
        passages = retriever.retrieve("tabel 3.c.2 publikasi")
        assert {p.document_name for p in passages[3:]} >= {"LED.docx"}
        assert [p.citation_number for p in passages] == list(range(1, len(passages) + 1))

    def test_a_longer_number_is_a_different_table(self, retriever):
        passages = retriever.retrieve("Tabel 3.C.2 Publikasi Penelitian")
        assert passages[0].section_label != "Tabel 3.C.20 Lain"

    def test_without_a_table_number_ranking_decides(self, retriever):
        passages = retriever.retrieve("publikasi penelitian dosen")
        assert passages[0].document_name == "LED.docx"
