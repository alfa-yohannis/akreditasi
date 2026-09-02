"""Command-line entry point for the Akreditasi RAG system.

Four subcommands, all sharing the same application objects:

    python main.py serve              start the web interface (default)
    python main.py ingest [--rebuild] build or refresh the index
    python main.py ask "pertanyaan"   answer one question in the terminal
    python main.py status             print index and backend status

Running the indexer from the terminal is useful for the first, longest pass:
it prints per-document progress and is unaffected by a browser tab closing.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import SETTINGS
from app.safety import SOURCE_GUARD

LOGGER = logging.getLogger("akreditasi")


class CommandLineInterface:
    """Parses arguments and dispatches to the matching command handler.

    Heavy modules are imported inside the handlers rather than at module level,
    so ``python main.py --help`` responds instantly instead of waiting for
    PyTorch to load.
    """

    def __init__(self) -> None:
        """Build the argument parser with one subparser per command."""
        self._parser = argparse.ArgumentParser(
            prog="main.py",
            description="Sistem RAG dokumen akreditasi (baca-saja atas direktori sumber).",
        )
        subparsers = self._parser.add_subparsers(dest="command")

        serve_parser = subparsers.add_parser("serve", help="Jalankan antarmuka web")
        serve_parser.add_argument("--host", default=SETTINGS.server_host)
        serve_parser.add_argument("--port", type=int, default=SETTINGS.server_port)
        serve_parser.add_argument(
            "--reload", action="store_true", help="Muat ulang otomatis saat kode berubah"
        )

        ingest_parser = subparsers.add_parser("ingest", help="Bangun atau perbarui indeks")
        ingest_parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Hapus indeks lama dan baca ulang seluruh dokumen",
        )

        ask_parser = subparsers.add_parser("ask", help="Ajukan satu pertanyaan")
        ask_parser.add_argument("question", help="Pertanyaan yang ingin dijawab")
        ask_parser.add_argument("--folder", default=None, help="Batasi ke satu folder utama")
        ask_parser.add_argument(
            "--backend", default=None, help="ollama atau claude_cli"
        )

        snapshot_parser = subparsers.add_parser(
            "snapshot",
            help="Ambil salinan halaman SINTA/Scholar yang dirujuk dokumen, lalu indeks",
        )
        snapshot_parser.add_argument(
            "--no-ingest",
            action="store_true",
            help="Hanya ambil salinan, jangan langsung mengindeks",
        )
        snapshot_parser.add_argument(
            "--http-only",
            action="store_true",
            help="Pakai HTTP biasa saja; lewati browser, login SINTA, dan PDDikti",
        )

        subparsers.add_parser("status", help="Tampilkan ringkasan indeks dan backend")

    def run(self, arguments: list[str]) -> int:
        """Execute the requested command and return a process exit code."""
        parsed = self._parser.parse_args(arguments)
        self._configure_logging()
        SOURCE_GUARD.verify_data_directory_isolation()

        command = parsed.command or "serve"
        handlers = {
            "serve": self._run_server,
            "ingest": self._run_ingest,
            "ask": self._run_ask,
            "status": self._run_status,
            "snapshot": self._run_snapshot,
        }
        return handlers[command](parsed)

    def _run_snapshot(self, parsed: argparse.Namespace) -> int:
        """Snapshot the external pages the corpus references, then index them."""
        from app.ingestion import INGESTION_PIPELINE
        from app.web_sources import WEB_SNAPSHOT_RUNNER

        print(f"\n  Menyimpan salinan ke: {WEB_SNAPSHOT_RUNNER.snapshot_directory}")
        if parsed.http_only:
            print("  Mode: HTTP biasa\n")
            outcome = WEB_SNAPSHOT_RUNNER.run(progress_callback=self._report_snapshot_progress)
        else:
            print("  Mode: browser (login SINTA, paginasi, PDDikti)")
            print("  Catatan: tutup Chrome dulu — porta debug tidak bisa dipakai bersamaan.\n")
            outcome = WEB_SNAPSHOT_RUNNER.run_with_browser(
                progress_callback=self._report_snapshot_progress
            )

        print("\n--- Ringkasan Salinan Eksternal ---")
        print(f"  Berhasil diambil : {outcome.fetched}")
        print(f"  Gagal            : {outcome.failed}")
        for message in outcome.errors:
            print(f"    - {message}")

        if outcome.fetched == 0:
            print("\n  Tidak ada halaman baru; indexing dilewati.")
            return 0
        if parsed.no_ingest:
            print("\n  Jalankan 'main.py ingest' untuk memasukkannya ke indeks.")
            return 0

        print("\n  Mengindeks salinan baru...")
        summary = INGESTION_PIPELINE.run()
        print(f"  Dokumen diindeks : {summary['documents_indexed']}")
        print(f"  Potongan dibuat  : {summary['chunks_created']}")
        return 0

    def _report_snapshot_progress(self, done: int, total: int, description: str) -> None:
        """Print one progress line per snapshotted page."""
        print(f"  [{done}/{total}] {description}")

    def _configure_logging(self) -> None:
        """Configure console logging for terminal use."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    def _run_server(self, parsed: argparse.Namespace) -> int:
        """Start the uvicorn server hosting the web interface."""
        import uvicorn

        self._print_access_banner(parsed.host, parsed.port)
        uvicorn.run(
            "app.server:create_application",
            factory=True,
            host=parsed.host,
            port=parsed.port,
            reload=parsed.reload,
            log_level="info",
        )
        return 0

    def _print_access_banner(self, host: str, port: int) -> None:
        """Show where the interface is reachable, and warn if it is exposed."""
        # Plain ASCII: the Windows console default code page mangles arrows.
        print(f"\n  Antarmuka web  ->  http://127.0.0.1:{port}")

        if host not in {"0.0.0.0", "::"}:
            print(f"  Hanya dapat diakses dari komputer ini (host={host}).")
        else:
            primary = self._primary_network_address()
            if primary:
                print(f"  Dari komputer lain ->  http://{primary}:{port}")
            others = [a for a in self._local_network_addresses() if a != primary]
            if others:
                print(f"  Alamat lain (adapter virtual/VPN): {', '.join(others)}")
            print(
                "\n  PERHATIAN: server terbuka ke jaringan tanpa autentikasi.\n"
                "  Siapa pun yang dapat menjangkau komputer ini bisa membaca\n"
                "  seluruh dokumen akreditasi yang terindeks.\n"
                "  Izinkan lewat firewall dengan (jalankan sebagai Administrator):\n"
                f"    New-NetFirewallRule -DisplayName 'Akreditasi RAG' "
                f"-Direction Inbound -LocalPort {port} -Protocol TCP -Action Allow -Profile Private"
            )

        print(f"\n  Sumber (baca-saja): {SETTINGS.source_directory}")
        print(f"  Data indeks       : {SETTINGS.data_directory}\n")

    def _primary_network_address(self) -> str | None:
        """Return the address colleagues should actually use.

        A machine with VMware, Hyper-V or a VPN reports several private
        addresses, and only one of them is on the real network. Opening a UDP
        socket towards a public address makes the operating system select the
        default route, and the local end of that socket is the interface that
        traffic genuinely leaves by. No packet is ever sent.
        """
        import socket

        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        except OSError:
            LOGGER.debug("Tidak dapat menentukan alamat utama", exc_info=True)
            return None
        finally:
            probe.close()

    def _local_network_addresses(self) -> list[str]:
        """Return this machine's other IPv4 addresses, excluding link-local ones.

        ``169.254.x.x`` is filtered out: it means the adapter never received a
        DHCP lease, so advertising it would send colleagues to a dead address.
        """
        import socket

        addresses: list[str] = []
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                address = info[4][0]
                if address.startswith(("127.", "169.254.")):
                    continue
                if address not in addresses:
                    addresses.append(address)
        except OSError:
            LOGGER.debug("Tidak dapat menentukan alamat jaringan lokal", exc_info=True)
        return addresses

    def _run_ingest(self, parsed: argparse.Namespace) -> int:
        """Run one indexing pass in the foreground with progress output."""
        from app.ingestion import INGESTION_PIPELINE

        LOGGER.info("Direktori sumber (baca-saja): %s", SETTINGS.source_directory)
        summary = INGESTION_PIPELINE.run(force_full_rebuild=parsed.rebuild)

        print("\n--- Ringkasan Indexing ---")
        print(f"  Berkas ditemukan : {summary['documents_total']}")
        print(f"  Diindeks         : {summary['documents_indexed']}")
        print(f"  Dilewati         : {summary['documents_skipped']}")
        print(f"  Gagal            : {summary['documents_failed']}")
        print(f"  Dihapus          : {summary['documents_removed']}")
        print(f"  Potongan dibuat  : {summary['chunks_created']}")
        print(f"  Halaman terpindai: {summary['scanned_pages_found']}")
        print(f"  Durasi           : {summary['elapsed_seconds']} detik")
        if summary["scanned_pages_found"]:
            print(
                "\n  Catatan: sejumlah halaman tidak memiliki lapisan teks. "
                "Pasang rapidocr-onnxruntime lalu set OCR_ENABLED=1 untuk membacanya."
            )
        return 1 if summary.get("error_message") else 0

    def _run_ask(self, parsed: argparse.Namespace) -> int:
        """Answer one question, streaming the result to standard output."""
        from app.embeddings import TextEmbedder  # noqa: F401  (warms the import chain)
        from app.ingestion import TEXT_EMBEDDER
        from app.rag import LANGUAGE_MODEL_FACTORY, QuestionRequest, RagEngine
        from app.retrieval import CROSS_ENCODER_RERANKER, HybridRetriever
        from app.storage import DOCUMENT_CATALOGUE, VECTOR_INDEX

        retriever = HybridRetriever(
            settings=SETTINGS,
            catalogue=DOCUMENT_CATALOGUE,
            vector_index=VECTOR_INDEX,
            embedder=TEXT_EMBEDDER,
            reranker=CROSS_ENCODER_RERANKER,
        )
        engine = RagEngine(SETTINGS, retriever, LANGUAGE_MODEL_FACTORY)
        request = QuestionRequest(
            question=parsed.question,
            folder_filter=parsed.folder,
            backend_name=parsed.backend,
        )

        print()
        citations: list[dict] = []
        for event in engine.stream_answer(request):
            if event.name == "token":
                print(event.payload["text"], end="", flush=True)
            elif event.name == "sources":
                citations = event.payload["citations"]
            elif event.name == "error":
                print(f"\n[GALAT] {event.payload['message']}", file=sys.stderr)
            elif event.name == "done":
                print(f"\n\n--- {event.payload.get('elapsed_seconds', 0)} detik ---")

        if citations:
            print("\nSumber:")
            for citation in citations:
                print(
                    f"  [{citation['citation_number']}] {citation['document_name']} "
                    f"— {citation['section_label']}"
                )
                print(f"       {citation['relative_path']}")
        return 0

    def _run_status(self, _parsed: argparse.Namespace) -> int:
        """Print a terminal summary of the index and backend availability."""
        from app.rag import LANGUAGE_MODEL_FACTORY
        from app.storage import DOCUMENT_CATALOGUE, VECTOR_INDEX

        statistics = DOCUMENT_CATALOGUE.collect_statistics()
        print("\n--- Status Indeks ---")
        print(f"  Sumber           : {SETTINGS.source_directory}")
        print(f"  Dokumen terindeks: {statistics['document_count']}")
        print(f"  Potongan         : {statistics['chunk_count']}")
        print(f"  Vektor           : {VECTOR_INDEX.size} × {VECTOR_INDEX.dimensions}")
        print(f"  Gagal dibaca     : {statistics['failed_count']}")
        print(f"  Terakhir diindeks: {statistics['last_indexed_at'] or 'belum pernah'}")

        print("\n--- Backend Model Bahasa ---")
        for backend_status in LANGUAGE_MODEL_FACTORY.check_all_statuses():
            marker = "OK  " if backend_status["is_available"] else "TIDAK SIAP"
            effort = backend_status.get("effort")
            model = backend_status["model_name"]
            if effort:
                model = f"{model} · effort {effort}"
            print(
                f"  [{marker}] {backend_status['backend_name']} "
                f"({model}) — {backend_status['detail']}"
            )
        print()
        return 0


if __name__ == "__main__":
    sys.exit(CommandLineInterface().run(sys.argv[1:]))
