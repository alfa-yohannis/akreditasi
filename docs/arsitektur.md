# Arsitektur

Dokumen ini menjelaskan bagaimana sistem disusun, bagaimana data mengalir dari
berkas di shared drive sampai menjadi jawaban bersitasi, serta **alasan** di
balik keputusan-keputusan rancangan yang tidak langsung terlihat jelas.

---

## Gambaran umum

```
┌────────────────────────────────────────────────────────────────────────┐
│  SUMBER (baca-saja)                                                    │
│  G:\Shared drives\...\2026          corpus-lokal\                      │
└───────────────┬───────────────────────────────┬────────────────────────┘
                │                               │
                ▼                               ▼
        ┌───────────────────────────────────────────────┐
        │  CorpusScanner                                │
        │  telusuri → saring → SourceDocument           │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  ExtractorRegistry                            │
        │  PDF / DOCX / XLSX / PPTX / HTML / CSV / ...  │
        │  → ExtractedSection (berlabel)                │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  TokenAwareChunker                            │
        │  → TextChunk (dengan tumpang tindih)          │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  TextEmbedder (BGE-M3, GPU)                   │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌──────────────────────────┬────────────────────┐
        │  DocumentCatalogue       │  VectorIndex       │
        │  SQLite + FTS5 (BM25)    │  NumPy .npz        │
        └──────────────────────────┴────────────────────┘
                            ▲
                            │  pertanyaan
        ┌───────────────────┴───────────────────────────┐
        │  HybridRetriever                              │
        │  vektor + kata kunci → RRF → cross-encoder    │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  RagEngine → PromptBuilder → Backend model     │
        │  → aliran SSE ke peramban                     │
        └───────────────────────────────────────────────┘
```

---

## Lapisan dan tanggung jawabnya

| Modul | Tanggung jawab |
|---|---|
| `config.py` | Membaca variabel lingkungan menjadi objek `Settings` bertipe |
| `safety.py` | Menegakkan akses baca-saja atas korpus |
| `documents.py` | Objek nilai yang dipakai bersama seluruh tahap |
| `extractors.py` | Mengubah setiap format berkas menjadi bagian teks berlabel |
| `ocr.py` | OCR opsional untuk halaman tanpa lapisan teks |
| `chunking.py` | Memotong bagian menjadi potongan yang dapat di-embed |
| `embeddings.py` | Encoder BGE-M3 dan penentuan perangkat |
| `reranking.py` | Tahap presisi cross-encoder |
| `storage.py` | Katalog SQLite dan indeks vektor NumPy |
| `ingestion.py` | Pemindai korpus dan pipeline inkremental |
| `retrieval.py` | Pencarian hibrida dan penggabungan peringkat |
| `language_models.py` | Backend Ollama dan tiga backend CLI agen |
| `rag.py` | Penyusunan prompt dan aliran jawaban |
| `answer_streams.py` | Jawaban dibuat di latar belakang; pembaca dapat menyambung kembali |
| `server.py` | Rute HTTP dan Server-Sent Events |

Ketergantungan mengalir **satu arah**, dari atas ke bawah daftar. `storage.py`
tidak tahu-menahu soal HTTP; `server.py` tidak pernah menyentuh SQLite langsung.

---

## Keputusan rancangan dan alasannya

### Mengapa bukan basis data vektor?

Korpus ini menghasilkan sekitar 15–25 ribu potongan. Pada 20.000 × 1024 dimensi
float32, seluruh matriks berukuran ±82 MB — muat seluruhnya di memori. Perkalian
titik menyeluruh atas matriks sebesar itu selesai dalam hitungan milidetik satu
digit, dengan recall **eksak**.

Chroma, FAISS, atau Qdrant akan menambah dependensi, menambah kerumitan
pemasangan di Windows, dan justru **mengurangi** recall (karena bersifat
aproksimatif) demi menghemat waktu yang tidak akan pernah terasa oleh pengguna.

Bila korpus kelak tumbuh melewati beberapa ratus ribu potongan,
`VectorIndex.search` adalah satu-satunya metode yang perlu diganti — antarmukanya
sudah berbentuk `search(vector, limit, allowed_ids) -> [(chunk_id, score)]`.

### Mengapa pencarian hibrida, bukan vektor saja?

Keduanya gagal pada kasus yang berbeda:

- Pencarian **vektor** memahami bahwa *"beban mengajar dosen"* dan *"EWMP"*
  merujuk hal yang sama, tetapi meleset pada pengenal persis.
- Pencarian **kata kunci** menemukan `SK-023/PRA/XI/2024` dengan tepat, tetapi
  buta terhadap parafrasa.

Dokumen akreditasi penuh keduanya: konsep bernarasi sekaligus nomor SK, NIDN, dan
angka tabel. Menghilangkan salah satu metode akan menghilangkan sebagian
pertanyaan yang wajar diajukan.

### Mengapa menggabungkan peringkat, bukan skor?

Kemiripan kosinus berada pada rentang sekitar 0–1; BM25 pada skala tak terbatas
yang bergantung statistik korpus. Penjumlahan berbobot dari keduanya akan perlu
disetel ulang setiap kali korpus berubah.

Reciprocal rank fusion hanya memakai **posisi** peringkat:

```
skor(d) = Σ  1 / (k + peringkat(d, daftar))
```

Konstanta `k` (bawaan 60) meredam pengaruh peringkat teratas, sehingga satu
metode tidak mendominasi hasil gabungan. Dokumen yang muncul di kedua daftar
naik di atas dokumen yang hanya disukai satu metode — persis perilaku yang
diinginkan.

### Mengapa setiap potongan tabel Word membawa judul dan kepala kolomnya?

Tabel panjang, misalnya Tabel 3.C.2 Publikasi Penelitian di LKPS, terpotong
menjadi puluhan potongan. Tanpa konteks, satu baris seperti
`<nama dosen> | <judul> | IB | | | V` tidak memuat kata "publikasi" maupun arti
kolomnya, sehingga tidak ditemukan oleh pertanyaan tentang publikasi.
`WordExtractor` karena itu membaca dokumen sesuai urutannya: setiap tabel
menjadi bagian tersendiri berlabel judulnya ("Tabel 3.C.2 Publikasi
Penelitian"), dan judul beserta baris kepala kolomnya diulang di awal setiap
potongan tabel itu. Judul itu *dipindahkan* dari teks sebelum tabel, bukan
disalin: potongan yang hanya berisi "Tabel 3.C.2 Publikasi Penelitian" lebih
cocok dengan pertanyaan yang menyebut tabel itu daripada baris-baris tabelnya
sendiri, dan model lalu menjawab dari judul tanpa data. Revisi ekstraksi
`.docx` tercatat di tanda tangan berkas (`SourceDocument.EXTRACTION_REVISIONS`),
sehingga indexing berikutnya membaca ulang semua `.docx` sekali saja, tanpa
mengulang OCR berkas PDF.

### Mengapa tabel yang disebut nomornya didahulukan?

Cross-encoder menilai baris tabel (`… | … | V`) lebih rendah daripada prosa
yang memakai kata yang sama, misalnya ringkasan di LED. Bila pertanyaan
menyebut tabel dengan nomor bertitik ("Tabel 3.C.2", "tabel 2.a.1"), potongan
tabel berlabel nomor itu mengisi hingga empat dari enam tempat: dari berkas
yang paling baru diubah lebih dulu (LKPS final, bukan draf berangka lama),
lalu menurut kedekatan makna. Dua tempat sisanya tetap dipilih reranker.
Nomor tanpa titik ("Tabel 6") tidak dipakai, karena tabel tanpa judul diberi
label "Tabel N" menurut urutannya.

### Mengapa salinan dokumen tidak mengisi dua tempat?

Arsip menyimpan beberapa versi satu dokumen (final, draf, referensi), sehingga
potongan terbaik untuk satu pertanyaan sering kali teks yang sama dari beberapa
berkas; pada tabel publikasi LKPS, keenamnya. Setelah reranking, potongan yang
sama atau hampir sama (≥ 90 % kata yang sama) dengan potongan yang sudah
terpilih dilewati, sehingga tempatnya diisi potongan lain.

### Mengapa hanya model penjawab yang ada di GPU?

Anggaran VRAM pada RTX 4060 (8 GB), diukur (`evaluation_results/`):

| | |
|---|---|
| Desktop Windows | ~0,6–1,1 GB |
| Qwen3 8B Q4_K_M, konteks 8.192 token | ~5,8 GB |
| BGE-M3 fp32 | ~2,3 GB |
| BGE-reranker-v2-m3, bila dimuat ke GPU | ~2,3 GB (fp32) |

Desktop dan model penjawab saja sudah memakai ±6,5 GB. Dulu BGE-M3 ikut tinggal
di GPU untuk meng-embed pertanyaan, sehingga totalnya ±8,7 GB. Windows tidak
menolak: kelebihannya dipinjamkan dari RAM sistem lewat PCIe (*Shared GPU
memory*), dan Ollama tetap melaporkan model "100 % GPU". Akibatnya model membaca
prompt 11× dan menulis 9× lebih lambat — jawaban median 35 detik, bukan 3,8.
Kini pertanyaan di-embed di CPU (±44 ms alih-alih ±11 ms, dengan kandidat yang
sama), BGE-M3 hanya memakai GPU selama indexing, dan reranker berjalan di CPU.

Reranker memakai OpenVINO int8: 24 kandidat dinilai dalam ±3,1 detik
(PyTorch: ±9,4 detik), dengan 95 % pilihan enam passage yang sama. Bila model
int8 belum dibuat (`main.py convert-reranker`), PyTorch dipakai otomatis.

### Mengapa Server-Sent Events, bukan WebSocket?

Lalu lintasnya satu arah dan berumur pendek: peramban mengirim satu pertanyaan
lalu membaca aliran sampai selesai. SSE menempuh HTTP biasa, memiliki cara baku
untuk melanjutkan aliran (`id` dan `Last-Event-ID`), dan tidak memerlukan
pembukuan koneksi. WebSocket hanya akan menambah kerumitan tanpa manfaat.

Sitasi dikirim **sebelum** token jawaban pertama, sehingga panel bukti dapat
digambar selagi model masih menulis — pengguna bisa mulai memeriksa sumber saat
bagian paling lambat dari permintaan sedang berjalan.

### Mengapa jawaban tidak dibuat di dalam respons HTTP?

Karena koneksi di ponsel mudah putus, dan jeda sunyi dalam satu jawaban bisa
panjang: pencarian beberapa detik, memuat model setelah server dimulai ulang
hampir satu menit, dan backend CLI pada effort tertinggi dapat berpikir
bermenit-menit sebelum kata pertama. Mengunci layar atau berpindah aplikasi di
tengah jeda itu memutus koneksi. Dulu jawaban dibuat di dalam respons, sehingga
putusnya koneksi ikut menghentikan pembuatannya: pengguna melihat sumber, lalu
*"network error"*, dan jawabannya tidak pernah tersimpan.

Kini setiap pertanyaan menjadi satu `AnswerJob` yang berjalan di thread
sendiri dan mencatat setiap peristiwa dengan nomor urut. Respons HTTP hanyalah
**pembaca** catatan itu:

```
POST /api/ask                          accepted {answer_id}, lalu peristiwa #1, #2, …
GET  /api/answers/{id}/events?after=N  peristiwa sesudah #N, lalu mengikuti yang baru
POST /api/answers/{id}/cancel          hentikan pembuatan jawaban
```

- Selama tidak ada peristiwa baru, server mengirim komentar `: keep-alive`
  setiap 10 detik agar koneksi tidak pernah tampak menganggur.
- Peramban yang kehilangan koneksi — atau yang tidak menerima apa pun selama
  35 detik, atau yang kembali ke halaman setelah lebih dari 15 detik sunyi —
  menyambung ke `/events?after=N` dan melanjutkan tepat dari peristiwa terakhir
  yang diterimanya. Selama halaman tersembunyi, percobaan ditunda sampai
  halaman terlihat lagi, karena ponsel memang menangguhkan jaringan saat itu.
- Jawaban tetap selesai dan tersimpan ke percakapan walaupun tidak ada yang
  membaca. Justru karena itu tombol **Hentikan** kini memanggil `/cancel`:
  menutup koneksi saja tidak lagi menghentikan server.
- Pekerjaan yang sudah selesai disimpan 30 menit (paling banyak 50) agar
  pembaca yang tertinggal masih dapat mengambil sisanya. Server yang dimulai
  ulang kehilangan semuanya; peramban lalu menyarankan bertanya kembali.

### Mengapa generator sinkron di dalam server asinkron?

`RagEngine.stream_answer` adalah generator biasa, bukan `async`. Generator itu
dijalankan di thread milik `AnswerJob`, dan pembaca SSE-nya juga generator
sinkron yang dijalankan Starlette di thread pool. Menunggu peristiwa berikutnya
atau panggilan model yang memblokir tidak pernah menahan event loop — tanpa
memaksa seluruh pipeline menjadi asinkron.

### Mengapa indexing inkremental?

Sekali jalan penuh atas 707 berkas memakan puluhan menit, hampir seluruhnya
untuk embedding. Tanda tangan `ukuran:mtime` per berkas membuat proses kedua
hanya membaca yang benar-benar berubah, dan selesai dalam hitungan detik.

### Rekonsiliasi dua penyimpanan

Baris potongan disimpan ke SQLite per dokumen, sedangkan arsip vektor hanya
di-flush setiap 25 dokumen. Bila proses terhenti di antara keduanya, dokumen
akan tercatat "terindeks" tetapi vektornya hilang — dan karena tanda tangannya
tidak berubah, setiap proses inkremental berikutnya akan melewatinya selamanya.

`find_documents_with_missing_vectors` membandingkan kedua penyimpanan di awal
setiap proses dan menghapus tanda tangan dokumen yang kehilangan vektor,
sehingga dokumen tersebut dibaca ulang. Lihat `tests/test_ingestion.py`
kelas `TestStoreReconciliation`.

---

## Model penegakan baca-saja

Setiap operasi berkas melewati `SourceTreeGuard`:

```
open_for_reading(path, mode)
    └─ tolak bila mode mengandung w / a / x / +

assert_writable(path)
    └─ normalkan symlink dan ".."
    └─ tolak bila berada di dalam SOURCE_DIR
```

Titik pemanggilannya:

| Pemanggil | Penjagaan |
|---|---|
| Setiap extractor | `open_for_reading` |
| `DocumentCatalogue.__init__` | `assert_writable(database_path)` |
| `VectorIndex.__init__` | `assert_writable(vector_store_path)` |
| Start server | `verify_data_directory_isolation()` |

Selain itu `openpyxl` dibuka dengan `read_only=True`, sehingga tidak pernah
memegang handle yang dapat menulis ke berkas Excel.

---

## Beberapa roots korpus

`CorpusRoot` memungkinkan lebih dari satu pohon direktori mengisi satu indeks.
Korpus utama memakai label kosong sehingga jalurnya terbaca persis seperti di
shared drive; setiap root tambahan memakai nama direktorinya sebagai awalan.

Awalan itu menjaga `document_id` tetap unik bila dua root kebetulan memuat
sub-jalur yang sama, sekaligus membuat asal sebuah sitasi terlihat di antarmuka.
Root tambahan yang tidak ada akan dilewati tanpa galat, sehingga rekan kerja
yang belum memiliki salinan lokal tetap dapat menjalankan indexing.
