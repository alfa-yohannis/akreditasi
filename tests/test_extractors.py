"""Tests for format-specific text extraction and the extractor registry."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.documents import SourceDocument
from app.extractors import (
    DelimitedTextExtractor,
    ExtractorRegistry,
    HtmlExtractor,
    JsonExtractor,
    PlainTextExtractor,
    SpreadsheetExtractor,
    WordExtractor,
)
from app.ingestion import FileTypeSniffer


def describe_file(file_path: Path, source_root: Path) -> SourceDocument:
    """Build a SourceDocument for a real file on disk."""
    file_status = file_path.stat()
    return SourceDocument(
        absolute_path=file_path,
        relative_path=str(file_path.relative_to(source_root)),
        display_name=file_path.name,
        extension=file_path.suffix.lower(),
        size_bytes=file_status.st_size,
        modified_timestamp=file_status.st_mtime,
    )


class TestPlainTextExtractor:
    """Reading .txt, .md and .py files."""

    def test_reads_utf8_content(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "catatan.txt"
        target.write_text("Beban mengajar dosen — EWMP", encoding="utf-8")

        outcome = PlainTextExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert outcome.succeeded
        assert "EWMP" in outcome.sections[0].text

    def test_falls_back_to_a_legacy_encoding(self, isolated_settings, guard):
        """Older exports on this corpus are Windows-1252, not UTF-8."""
        target = isolated_settings.source_directory / "lama.txt"
        target.write_bytes("Ringkasan – capaian".encode("cp1252"))

        outcome = PlainTextExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert outcome.succeeded
        assert "Ringkasan" in outcome.sections[0].text

    def test_empty_file_yields_no_sections(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "kosong.txt"
        target.write_text("", encoding="utf-8")

        outcome = PlainTextExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert outcome.sections == []

    def test_missing_file_becomes_an_error_outcome(self, isolated_settings, guard):
        """An unreadable file must not abort a whole ingest run."""
        missing = isolated_settings.source_directory / "tidak-ada.txt"
        document = SourceDocument(
            absolute_path=missing, relative_path="tidak-ada.txt",
            display_name="tidak-ada.txt", extension=".txt",
            size_bytes=10, modified_timestamp=0.0,
        )
        outcome = PlainTextExtractor(isolated_settings, guard).extract(document)
        assert outcome.succeeded is False
        assert outcome.error_message is not None


class TestJsonExtractor:
    """Reading .json files."""

    def test_pretty_prints_valid_json(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "data.json"
        target.write_text(json.dumps({"prodi": "MTI", "tahun": 2026}), encoding="utf-8")

        outcome = JsonExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert "prodi" in outcome.sections[0].text

    def test_malformed_json_is_indexed_as_raw_text(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "rusak.json"
        target.write_text('{"prodi": "MTI"', encoding="utf-8")

        outcome = JsonExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert outcome.succeeded
        assert "MTI" in outcome.sections[0].text


class TestDelimitedTextExtractor:
    """Reading .csv files."""

    def test_preserves_rows_and_columns(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "dosen.csv"
        target.write_text("Nama,EWMP\nGufron,13\nAndi,11\n", encoding="utf-8")

        outcome = DelimitedTextExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        text = outcome.sections[0].text
        assert "Nama\tEWMP" in text
        assert "Gufron\t13" in text

    def test_blank_rows_are_dropped(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "renggang.csv"
        target.write_text("a,b\n\n,\nc,d\n", encoding="utf-8")

        outcome = DelimitedTextExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert outcome.sections[0].text.count("\n") == 1


class TestHtmlExtractor:
    """Reading .html files."""

    def test_strips_markup_and_scripts(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "halaman.html"
        target.write_text(
            "<html><head><style>p{color:red}</style></head>"
            "<body><script>var x=1;</script><p>Capaian pembelajaran lulusan</p></body></html>",
            encoding="utf-8",
        )
        outcome = HtmlExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        text = outcome.sections[0].text
        assert "Capaian pembelajaran lulusan" in text
        assert "var x" not in text
        assert "color:red" not in text


class TestSpreadsheetExtractor:
    """Reading .xlsx workbooks."""

    @pytest.fixture
    def workbook_path(self, isolated_settings):
        """A two-sheet workbook written with openpyxl."""
        import openpyxl

        workbook = openpyxl.Workbook()
        first_sheet = workbook.active
        first_sheet.title = "Dosen"
        first_sheet.append(["Nama", "EWMP"])
        first_sheet.append(["Gufron", 13])

        second_sheet = workbook.create_sheet("Kosong")
        second_sheet.append([None, None])

        target = isolated_settings.source_directory / "beban.xlsx"
        workbook.save(target)
        return target

    def test_each_sheet_becomes_its_own_section(self, isolated_settings, guard, workbook_path):
        outcome = SpreadsheetExtractor(isolated_settings, guard).extract(
            describe_file(workbook_path, isolated_settings.source_directory)
        )
        assert len(outcome.sections) == 1
        assert outcome.sections[0].label == "Sheet: Dosen"

    def test_cells_are_rendered_tab_separated(self, isolated_settings, guard, workbook_path):
        outcome = SpreadsheetExtractor(isolated_settings, guard).extract(
            describe_file(workbook_path, isolated_settings.source_directory)
        )
        assert "Nama\tEWMP" in outcome.sections[0].text
        assert "Gufron\t13" in outcome.sections[0].text

    def test_reading_leaves_the_workbook_byte_identical(
        self, isolated_settings, guard, workbook_path
    ):
        """openpyxl must never write back to a corpus file."""
        original_bytes = workbook_path.read_bytes()
        SpreadsheetExtractor(isolated_settings, guard).extract(
            describe_file(workbook_path, isolated_settings.source_directory)
        )
        assert workbook_path.read_bytes() == original_bytes


class TestWordExtractor:
    """Reading .docx documents."""

    @pytest.fixture
    def word_path(self, isolated_settings):
        """A document containing a paragraph and a table."""
        import docx

        document = docx.Document()
        document.add_paragraph("Visi program studi adalah unggul di bidang informatika.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Nama"
        table.cell(0, 1).text = "Jabatan"
        table.cell(1, 0).text = "Gufron"
        table.cell(1, 1).text = "Lektor"

        target = isolated_settings.source_directory / "visi.docx"
        document.save(target)
        return target

    def test_extracts_paragraph_text(self, isolated_settings, guard, word_path):
        outcome = WordExtractor(isolated_settings, guard).extract(
            describe_file(word_path, isolated_settings.source_directory)
        )
        assert "Visi program studi" in outcome.sections[0].text

    def test_flattens_tables_into_pipe_separated_rows(self, isolated_settings, guard, word_path):
        outcome = WordExtractor(isolated_settings, guard).extract(
            describe_file(word_path, isolated_settings.source_directory)
        )
        assert "Gufron | Lektor" in outcome.sections[1].text

    def extract(self, isolated_settings, guard, build) -> list:
        """Save the document ``build`` fills in, and return its sections."""
        import docx

        document = docx.Document()
        build(document)
        target = isolated_settings.source_directory / "lkps.docx"
        document.save(target)
        outcome = WordExtractor(isolated_settings, guard).extract(
            describe_file(target, isolated_settings.source_directory)
        )
        assert outcome.succeeded
        return outcome.sections

    @staticmethod
    def add_table(document, rows: list[list[str]]):
        """Append a table holding ``rows``."""
        table = document.add_table(rows=len(rows), cols=len(rows[0]))
        for row_index, cells in enumerate(rows):
            for column_index, text in enumerate(cells):
                table.cell(row_index, column_index).text = text
        return table

    PUBLICATIONS = [
        ["Nama Dosen", "Judul Publikasi", "Tahun"],
        ["Dosen A", "Judul Publikasi Pertama", "2025"],
        ["Dosen B", "Judul Publikasi Kedua", "2024"],
    ]

    def test_tables_stay_in_reading_order(self, isolated_settings, guard):
        def build(document):
            document.add_paragraph("Bagian tiga membahas penelitian dosen.")
            document.add_paragraph("Tabel 3.B.2 Publikasi Ilmiah DTPR")
            self.add_table(document, self.PUBLICATIONS)
            document.add_paragraph("Bagian empat membahas pengabdian kepada masyarakat.")

        sections = self.extract(isolated_settings, guard, build)
        assert [section.label for section in sections] == [
            "Dokumen", "Tabel 3.B.2 Publikasi Ilmiah DTPR", "Dokumen"
        ]
        assert "penelitian dosen" in sections[0].text
        assert "Dosen A | Judul Publikasi Pertama | 2025" in sections[1].text
        assert "pengabdian" in sections[2].text

    def test_the_caption_leaves_the_prose_before_the_table(self, isolated_settings, guard):
        """Left in the prose, a caption alone outranks the table's own rows."""
        def build(document):
            document.add_paragraph("Bagian tiga membahas penelitian dosen.")
            document.add_paragraph("Tabel 3.B.2 Publikasi Ilmiah DTPR")
            self.add_table(document, self.PUBLICATIONS)

        prose = self.extract(isolated_settings, guard, build)[0]
        assert prose.text == "Bagian tiga membahas penelitian dosen."

    def test_caption_and_header_row_become_the_context(self, isolated_settings, guard):
        def build(document):
            document.add_paragraph("Tabel 3.B.2 Publikasi Ilmiah DTPR")
            self.add_table(document, self.PUBLICATIONS)

        sections = self.extract(isolated_settings, guard, build)
        assert len(sections) == 1
        assert sections[0].context == (
            "Tabel 3.B.2 Publikasi Ilmiah DTPR\nNama Dosen | Judul Publikasi | Tahun"
        )
        assert "Nama Dosen" not in sections[0].text

    def test_a_note_between_caption_and_table_goes_with_the_caption(self, isolated_settings, guard):
        def build(document):
            document.add_paragraph("Tabel 3.A.2 Penelitian DTPR")
            document.add_paragraph("(dalam juta rupiah)")
            self.add_table(document, self.PUBLICATIONS)

        table_section = self.extract(isolated_settings, guard, build)[0]
        assert table_section.label == "Tabel 3.A.2 Penelitian DTPR"
        assert table_section.context.startswith("Tabel 3.A.2 Penelitian DTPR\n(dalam juta rupiah)\n")

    def test_a_sentence_about_a_table_is_not_its_caption(self, isolated_settings, guard):
        sentence = "Tabel 3 menunjukkan " + "bahwa capaian penelitian meningkat setiap tahun " * 8 + "."

        def build(document):
            document.add_paragraph(sentence)
            self.add_table(document, self.PUBLICATIONS)

        sections = self.extract(isolated_settings, guard, build)
        assert sections[0].text.startswith("Tabel 3 menunjukkan")
        assert sections[1].label == "Tabel 1"

    def test_a_heading_before_the_table_serves_as_its_caption(self, isolated_settings, guard):
        def build(document):
            document.add_heading("Publikasi Ilmiah Dosen", level=2)
            self.add_table(document, self.PUBLICATIONS)

        assert self.extract(isolated_settings, guard, build)[0].label == "Publikasi Ilmiah Dosen"

    def test_the_caption_of_an_empty_table_stays_in_the_text(self, isolated_settings, guard):
        def build(document):
            document.add_paragraph("Tabel 9 Rencana Kegiatan")
            document.add_table(rows=2, cols=2)

        sections = self.extract(isolated_settings, guard, build)
        assert [section.text for section in sections] == ["Tabel 9 Rencana Kegiatan"]

    def test_a_sentence_before_the_table_is_not_a_caption(self, isolated_settings, guard):
        def build(document):
            document.add_paragraph("Data berikut diambil dari SISTER.")
            self.add_table(document, self.PUBLICATIONS)

        table_section = self.extract(isolated_settings, guard, build)[1]
        assert table_section.label == "Tabel 1"
        assert table_section.context == "Nama Dosen | Judul Publikasi | Tahun"

    def test_a_title_row_spanning_the_table_names_it(self, isolated_settings, guard):
        def build(document):
            table = self.add_table(document, [["", "", ""]] + self.PUBLICATIONS)
            title = table.cell(0, 0).merge(table.cell(0, 2))
            title.text = "Publikasi Ilmiah DTPR"

        table_section = self.extract(isolated_settings, guard, build)[0]
        assert table_section.label == "Publikasi Ilmiah DTPR"
        assert table_section.context == (
            "Publikasi Ilmiah DTPR\nNama Dosen | Judul Publikasi | Tahun"
        )

    def test_a_grouped_header_keeps_both_header_rows(self, isolated_settings, guard):
        def build(document):
            table = self.add_table(document, [
                ["Nama", "Jumlah Judul", ""],
                ["", "TS-1", "TS"],
                ["Dosen A", "2", "3"],
            ])
            table.cell(0, 1).merge(table.cell(0, 2)).text = "Jumlah Judul"

        table_section = self.extract(isolated_settings, guard, build)[0]
        assert table_section.context == "Nama | Jumlah Judul | Jumlah Judul\n | TS-1 | TS"
        assert table_section.text == "Dosen A | 2 | 3"

    def test_a_one_row_table_keeps_its_row_as_text(self, isolated_settings, guard):
        def build(document):
            self.add_table(document, [["Ketua Program Studi", "Dosen A"]])

        table_section = self.extract(isolated_settings, guard, build)[0]
        assert table_section.text == "Ketua Program Studi | Dosen A"
        assert table_section.context == ""


class TestExtractorRegistry:
    """Routing documents to the right extractor."""

    @pytest.fixture
    def registry(self, isolated_settings, guard):
        return ExtractorRegistry(isolated_settings, guard)

    @pytest.mark.parametrize(
        "extension", [".pdf", ".docx", ".xlsx", ".pptx", ".html", ".csv", ".json", ".txt", ".md"]
    )
    def test_supports_every_corpus_format(self, registry, extension):
        assert registry.supports(extension) is True

    def test_reports_unsupported_formats(self, registry):
        assert registry.supports(".mov") is False

    def test_extension_matching_is_case_insensitive(self, registry):
        assert registry.supports(".PDF") is True

    def test_unsupported_document_yields_a_described_failure(self, registry, isolated_settings):
        document = SourceDocument(
            absolute_path=isolated_settings.source_directory / "video.mov",
            relative_path="video.mov", display_name="video.mov",
            extension=".mov", size_bytes=1, modified_timestamp=0.0,
        )
        outcome = registry.extract(document)
        assert outcome.succeeded is False
        assert "extractor" in outcome.error_message.lower()


class TestFileTypeSniffer:
    """Recovering documents that lost their file extension."""

    def test_detects_a_pdf_by_magic_bytes(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "tanpa-ekstensi"
        target.write_bytes(b"%PDF-1.7\n" + b"0" * 4096)
        assert FileTypeSniffer(guard).detect_extension(target, target.stat().st_size) == ".pdf"

    def test_detects_a_jpeg(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "sertifikat"
        target.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 4096)
        assert FileTypeSniffer(guard).detect_extension(target, target.stat().st_size) == ".jpg"

    def test_distinguishes_docx_from_xlsx_inside_a_zip(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "dokumen-tanpa-ekstensi"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("word/document.xml", "<xml/>")
            archive.writestr("padding", "0" * 4096)
        assert FileTypeSniffer(guard).detect_extension(target, target.stat().st_size) == ".docx"

    def test_ignores_tiny_files(self, isolated_settings, guard):
        """A few bytes cannot be a real document; skip the read entirely."""
        target = isolated_settings.source_directory / "kecil"
        target.write_bytes(b"%PDF")
        assert FileTypeSniffer(guard).detect_extension(target, target.stat().st_size) is None

    def test_returns_none_for_unrecognised_content(self, isolated_settings, guard):
        target = isolated_settings.source_directory / "acak"
        target.write_bytes(b"\x01\x02\x03\x04" + b"0" * 4096)
        assert FileTypeSniffer(guard).detect_extension(target, target.stat().st_size) is None
