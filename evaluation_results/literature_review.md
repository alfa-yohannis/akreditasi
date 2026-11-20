# Faster stages for a local Indonesian RAG pipeline: literature and benchmark review

Compiled 2026-11-20. Scope: stages 1, 2, 3, 5 and 6 of the accreditation-document RAG pipeline (bge-m3 → NumPy cosine → FTS5 BM25 → RRF → bge-reranker-v2-m3 on CPU → Qwen3 8B Q4_K_M in Ollama), on one Windows 11 PC (i7-14700F, 32 GB RAM, RTX 4060 8 GB).

## How the evidence was gathered

- I only report numbers from sources I opened in this session: papers (arXiv HTML/PDF), model cards, the MTEB results repository, the SEA-HELM leaderboard, library docs and release notes. Anything I calculated myself is marked **[estimate]**. Anything I could not find is marked **not found**. Points I am unsure of are marked **[uncertain]**.
- **Indonesian retrieval numbers (MTEB).** For a like-for-like comparison of embedding models I read the per-language JSON files in the official `embeddings-benchmark/results` GitHub repository [R10] and extracted the Indonesian subset (`id`) of `MIRACLRetrievalHardNegatives` (dev split, main score = nDCG@10). This is the MIRACL task used in MMTEB(Multilingual). I also extracted `BelebeleRetrieval` (`ind_Latn-ind_Latn`, test) and `MIRACLReranking` (`id`, dev). Belebele is nearly saturated (80–99), so it separates models poorly. I give MIRACL-HN more weight.
- **Indonesian LLM numbers (SEA-HELM).** I took these from the data embedded in the SEA-HELM leaderboard page for Indonesian [R30] ("Data last updated Nov 20, 2026"). The page reports the overall Indonesian score, competency scores and task scores, each with 95% bootstrap CIs. The "QA" task is Indonesian extractive QA (metric ChrF++). It is the task closest to RAG answering.
- MKQA has no Indonesian split (it has Malay). The bge-m3 paper's MKQA table confirms this [R1]. So MKQA numbers below are multilingual averages only.

---

## 0. Where the latency goes

### 0.1 Literature

| Source | Setting | Finding |
|---|---|---|
| Shen et al., arXiv:2412.11854 [R41] | 100M-chunk datastore, FAISS on 32 Xeon cores; Gemma 2 9B on an A6000 Ada (vLLM) | Retrieval was ~41% of end-to-end latency and 45–47% of TTFT in the introduction; Takeaway 1 says "nearly 35%" of TTFT, so the paper is internally inconsistent. RAG stages raised TTFT from 495 ms to 965 ms. This applies to **very large** datastores. The authors call 100M chunks "relatively small", and throughput fell up to 20× from 1M to 100M chunks. |
| Feric et al., *RAGMark*, arXiv:2609.05760 [R42] | 2× A100-40GB; Llama-3 8B; MiniLM-L6 cross-encoder | "LLM TTFT and decode dominate end-to-end latency across all models and retrieval depths." Retrieval plus embedding stayed under 35 ms at top-k=10. Reranking overhead "scales linearly with retrieval depth, from ∼7 ms at topk=1 to ∼24 ms at topk=10". Reranking cut Llama-3-8B TTFT from 322 ms to 165 ms (−49%) because fewer passages reached the prompt. |
| Jiang et al., *RAGO*, arXiv:2503.14649 [R43] | RAG serving optimiser | Bottlenecks shift between retrieval and inference across RAG variants. Reports up to 2× QPS/chip and −55% TTFT versus LLM-system extensions. |
| Jacob et al., *Drowning in Documents*, arXiv:2411.11767 (ReNeuIR @ SIGIR 2025) [R44] | Scaling reranker depth | Rerankers improve at first as they score more documents, then "gradually decline" and can fall below earlier quality. Deeper reranking is not free quality. |

### 0.2 What this means for this pipeline [estimate]

The corpus has 22,944 passages, roughly 4–5 orders of magnitude smaller than the datastores where retrieval dominates [R41]. The measured dense (8 ms) and FTS5 (4 ms) times already match the "<35 ms" regime in RAGMark [R42]. The rough budget below is my own arithmetic. Measure it with Ollama's `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` and `load_duration` response fields [R35c].

| Stage | Time | Basis |
|---|---|---|
| 1 Query embed (bge-m3, GPU) | likely tens of ms or less | not found for this GPU; measure |
| 2 Dense search | 8 ms | measured by you |
| 3 FTS5 BM25 | 4 ms | measured by you |
| 5 Rerank 24 × ≤512 tokens, XLM-R-large on CPU | probably the largest non-LLM cost (order of seconds) **[estimate]** | not found; measure. The ST CPU benchmark only tested batch sizes 2 and 4 for bge-reranker-v2-m3 [R22b]. |
| 6a Prefill, ~14,000 chars (≈3–4.5k tokens) | ≈1.5–2.5 s **[estimate]** | llama.cpp scoreboard: RTX 4060 Ti 8 GB, Llama-2-7B Q4_0, pp512 = 3,395 t/s (FA off) / 3,803 t/s (FA on) [R36b]. The RTX 4060 has 3,072 vs 4,352 CUDA cores and the same 128-bit GDDR6 bus [R46]. |
| 6b Decode, up to 1,536 tokens | ≈30–38 s worst case at ~40–50 t/s **[estimate]**; ~10 s for a 500-token answer | Same scoreboard: tg128 = 63.9–64.0 t/s on a 4060 Ti 8 GB for a 7B Q4_0 [R36b]. Qwen3-8B Q4_K_M is a bigger file (5.2 GB in Ollama [R35d]), and decode speed scales roughly inversely with weight bytes. |

**Conclusion.** Generation decode dominates, prefill comes second, and CPU reranking third. Stages 1–3 together are at most a few percent. Most of the attainable speed-up is in stage 6 (fewer output tokens, a smaller or faster model, speculative decoding) and stage 5 (int8 or GPU reranking, fewer candidates). Stages 2–3 can be tidied, but they will not change user-visible latency.

---

## 1. Stage 1 — query embedding

### 1.1 Indonesian evidence for candidate embedders

MIRACL-HN id = MTEB `MIRACLRetrievalHardNegatives`, dev, `id` subset, nDCG@10, from the MTEB results repository [R10]. "Paper id" = MIRACL dev Indonesian nDCG@10 as reported in the model's own paper. Parameter counts come from the Hugging Face Hub API (safetensors total), unless marked "card".

| Model (HF ID) | Params | Dim | Licence | MIRACL-HN id | Belebele id | Other Indonesian evidence | Notes for ST 6 / Windows |
|---|---|---|---|---|---|---|---|
| **BAAI/bge-m3** (current) | 568M (Arctic card [R21]) | 1024 | MIT | **56.02** | 93.81 | MIRACL dev id: dense 56.1, sparse 48.9, dense+sparse 58.3, all 59.0 [R1]. SEA-BED ind 78.09 [R8]. | Works. If you ever indexed with ST 5.5.0–5.6.0 + flash attention, re-encode; the XLM-R position bug was fixed in 5.6.1 [R22d]. |
| Qwen/Qwen3-Embedding-0.6B | 596M | 1024 (MRL 32–1024) | Apache-2.0 | 52.43 | 91.15 | SEA-BED ind 75.60 [R8]; MTEB(Multilingual) Retrieval 64.64 vs bge-m3 54.60 (Qwen card [R3]) | ST ≥2.7. Needs an instruction prompt on queries (`prompt_name="query"`); the card says omitting it costs about 1–5% [R3]. |
| Qwen/Qwen3-Embedding-4B | ~4B | 2560 | Apache-2.0 | 56.77 | 95.06 | — | Too large to beat bge-m3 on speed. |
| google/embeddinggemma-300m | 303M | 768 (MRL 512/256/128) | Gemma terms (gated) | 51.74 | 94.45 | SEA-Embedding paper: SEA-BED ind 0.802, best in the ≤300M group [R9]. MTEB Multilingual v2 mean 61.15 at 768d, 59.68 at 256d [R5]. | **No float16 activations** (use fp32/bf16) [R5]. Query and document prompts are required. |
| microsoft/harrier-oss-v1-270m | 268M | 640 | MIT | 50.54 | 89.26 | SEA-BED ind 0.779 [R9]; MTEB v2 (multilingual) 66.5 (card) [R11] | Released 2026-03-30 (Hub). Decoder (Qwen3 architecture), last-token pooling, instruction prompt on queries [R11]. |
| microsoft/harrier-oss-v1-0.6b | 596M | 1024 | MIT | 53.29 | 92.79 | SEA-BED ind 0.809 [R9]; MTEB v2 69.0 [R11] | Same size as bge-m3, so no speed gain. |
| jinaai/jina-embeddings-v5-text-small | 677M (card) | 1024 (MRL) | **CC BY-NC 4.0** | 52.79 | 93.89 | SEA-BED ind 0.764 ("jina-embeddings-v5") [R9] | `trust_remote_code`, requires peft [R12]. |
| jinaai/jina-embeddings-v5-text-nano | 212M (Hub) | 768 (config) | **CC BY-NC 4.0** | 52.23 | 93.85 | — | `trust_remote_code`. |
| jinaai/jina-embeddings-v3 | 572M | 1024 | **CC BY-NC 4.0** | 49.42 | 92.47 | SEA-BED ind 77.35 [R8] | — |
| intfloat/multilingual-e5-small | 118M | 384 | MIT | 50.93 | 91.50 | MIRACL dev id 50.7, R@100 86.2 [R6]. Mr. TyDi id MRR@10 63.2 (base 64.9, large 68.5, BM25 38.2) [R6b]. | Plain BERT; `query:`/`passage:` prefixes. 512-token limit (config). |
| intfloat/multilingual-e5-base | 278M | 768 | MIT | 51.19 | 94.00 | MIRACL dev id 51.1 [R6] | — |
| intfloat/multilingual-e5-large-instruct | 560M | 1024 | MIT | 46.35 **[uncertain]** | 94.00 | MIRACL dev id 52.1 in its own paper [R6]; SEA-BED ind 79.50 [R8] | The MTEB run is far below the paper's figure, probably a prompt difference. Same size as bge-m3. |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 568M | 1024 | Apache-2.0 | 52.36 | 92.23 | — | Same size as bge-m3. |
| Snowflake/snowflake-arctic-embed-m-v2.0 | 305M | 768 (MRL 256) | Apache-2.0 | 48.51 | 83.60 | — | `trust_remote_code` (gte-multilingual-base backbone) [R21]. |
| Alibaba-NLP/gte-multilingual-base | 305M | 768 (elastic 128–768) | Apache-2.0 | 50.47 | n/a | MIRACL id (mGTE paper, Table 20): dense 50.3, dense+sparse 56.4 [R7] | `trust_remote_code`. The card claims "10x" faster inference than decoder models such as gte-Qwen2-1.5B [R7b]. |
| ibm-granite/granite-embedding-278m-multilingual | 278M | 768 | Apache-2.0 | 47.51 | 84.72 | **Indonesian is not among its 12 supported languages** [R20] | — |
| ibm-granite/granite-embedding-107m-multilingual | 107M | 384 | Apache-2.0 | 46.04 | 81.87 | Indonesian not supported [R20] | Card: "twice as fast as other models with similar embedding dimensions". |
| aisingapore/SEA-LION-ModernBERT-Embedding-300M | 312M | see note | MIT | not in MTEB repo | — | SEA-BED ind 0.787 (SEA-Embedding paper) [R9] | Released 2026-03. The card mentions only an `STS` prompt, and its stated dims are inconsistent (768 d_model vs 1024 output) [R9b]. **[uncertain]** |
| aisingapore/SEA-LION-E5-Embedding-600M | 560M | 1024 | MIT | not in MTEB repo | — | SEA-BED ind 0.808, best overall SEA-BED avg 0.800 [R9] | XLM-R large size (512 positions), so no speed gain over bge-m3. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (reference only) | 118M | 384 | Apache-2.0 | 28.18 | 80.22 | — | Shows that small, old models collapse on Indonesian retrieval. |

**Reading the table.** No model at or below bge-m3's size beats it on Indonesian MIRACL. The best ≤600M alternatives (harrier-0.6b 53.3, jina-v5-small 52.8, Qwen3-Emb-0.6B 52.4) are 3–4 nDCG points lower and no smaller. The only real size reductions are embeddinggemma-300m (−4.3 points, about half the parameters), harrier-270m (−5.5) and mE5-small (−5.1, one fifth of the parameters). Embedding a single short query is cheap anyway (§0.2), and switching models forces a full re-index of 22,944 passages. So the evidence says to **keep bge-m3 and optimise how it runs**.

### 1.2 Techniques (same model, faster or less VRAM)

All numbers come from the sentence-transformers efficiency benchmark [R22a]. GPU was an RTX 3090 and CPU an i7-13700K (same Raptor Lake generation as your i7-14700F). Models: MiniLM-L6, bge-base, mxbai-large and, on GPU, BAAI/bge-m3. Quality was measured on English NanoBEIR + STSb.

| Technique | Speed-up vs fp32 | Quality ratio | Notes |
|---|---|---|---|
| GPU fp16 (`model_kwargs={"torch_dtype": "float16"}`) | 2.92× | 100.00% | Easiest win if bge-m3 still runs in fp32. |
| GPU fp16 + FlashAttention-2 with unpadding | 3.87× | 100.39% | Needs `kernels` or `flash-attn`; Windows availability **[uncertain]**. Gains come from mixed-length batches, which matter little for a single query. |
| GPU ONNX-O4 (fp16) | 2.20× | 99.99% | No better than plain fp16. |
| CPU OpenVINO int8 (`export_static_quantized_openvino_model`) | 2.84× | 99.37% | Recommended in the ST flowchart for CPUs when "a small accuracy loss is acceptable". Moves the embedder off the GPU and frees ~1.1 GB VRAM (568M × 2 bytes, **[estimate]**) for the LLM or reranker. |
| CPU ONNX int8 (`export_dynamic_quantized_onnx_model`) | 1.92× | 99.58% | — |
| llama.cpp Q4_K_M | — | — | "BGE-M3 Q4 is excluded because its embeddings failed the agreement check against FP32" [R22a]. **Avoid 4-bit bge-m3.** |
| `model.compile(mode="reduce-overhead")` at batch size 1 | ~3.0× (bge-small, bf16, RTX 3090) | — | Works since ST 5.7.0. CUDA graphs need fixed padding lengths [R22c]. Measure on bge-m3. |

Caveat: an int8 query encoder against an fp32/fp16-encoded corpus is asymmetric. The ST benchmark used matching artifacts, so validate on your own question set before adopting it.

Also worth testing, for quality rather than speed: bge-m3's **sparse (lexical) output comes from the same forward pass**. Dense+sparse lifts MIRACL id from 56.1 to 58.3 [R1], and it could replace FTS5 (stage 3). I have not verified support for the bge-m3 sparse head in ST 6 **[uncertain]**; FlagEmbedding is the reference implementation.

### 1.3 Lightweight models from Chinese labs

| Model (lab) | Indonesian evidence | Verdict |
|---|---|---|
| BAAI/bge-m3 (BAAI) | Best ≤600M on MIRACL-HN id (56.02) | Keep. |
| BAAI/bge-multilingual-gemma2 (BAAI), 9.24B, Gemma licence | SEA-BED ind 79.93 [R8]. Indonesian MTEB subsets not found in the results repo. | Far too large for a query encoder on 8 GB alongside the LLM. |
| BAAI/bge-small-en-v1.5, BAAI/bge-small-zh-v1.5, bge-base-en-v1.5 (BAAI) | Hub language tags are `en` or `zh` only (confirmed). bge-small-en-v1.5 scores MIRACL-HN id **20.38** and Belebele id 54.17 [R10]. | **Confirmed unsuitable for Indonesian.** |
| Qwen/Qwen3-Embedding-0.6B / 4B / 8B (Alibaba Qwen) | 52.43 / 56.77 / 56.80 MIRACL-HN id [R10] | 0.6B is the same size as bge-m3 and weaker on Indonesian. |
| Alibaba-NLP/gte-multilingual-base (Alibaba Tongyi) | 50.47 [R10]; mGTE paper id dense 50.3 [R7] | About half of bge-m3's size but −5.5 points. |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | 51.48 [R10] | Larger and weaker. |
| KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 (HIT Shenzhen), 494M, Apache-2.0 | Used as the first stage in the Lychee/KaLM-Reranker paper: MIRACL id 49.89 (full MIRACL, top-100 setting) [R17]. Not in the MTEB repo's Indonesian files. | Weaker than bge-m3. |
| TencentBAC/Conan-embedding-v1 (Tencent) | Hub language `zh`, CC BY-NC 4.0 | Chinese-only, unsuitable. |
| TencentBAC/Conan-embedding-v2 (Tencent) | Hub languages `en`, `zh`; card claims Chinese–English cross-lingual retrieval | No Indonesian evidence found. |
| NovaSearch/stella_en_400M_v5 | English model (name and card); no language tags | No Indonesian evidence found. |
| codefuse-ai/F2LLM-0.6B (Ant Group) | Hub language `en`; Indonesian MTEB subsets not found | No Indonesian evidence found. |

### 1.4 Stage 1 shortlist (benchmark locally)

1. **`BAAI/bge-m3`, unchanged weights.** Run fp16 on GPU (2.92× over fp32 [R22a]) or OpenVINO int8 on CPU (2.84× over CPU fp32, 99.37% quality [R22a]) to free VRAM. No re-index is needed for fp16. Indonesian: 56.02 MIRACL-HN id, the best ≤600M.
2. **`google/embeddinggemma-300m`.** Half the parameters, 768-d (or 256-d MRL, −1.5 MMTEB points [R5]). MIRACL-HN id 51.74; best ≤300M on SEA-BED Indonesian (0.802) [R9]. Gemma licence; no fp16.
3. **`intfloat/multilingual-e5-small`.** 118M, 384-d, MIT. MIRACL id 50.7 [R6]. Use it only if the CPU-only footprint must be minimal.
   - MIT-licensed alternative to item 2: `microsoft/harrier-oss-v1-270m` (50.54).

---

## 2. Stage 2 — dense vector search (22,944 × 1024)

Evidence and options:

- **Exact search is the right regime.** HNSW and IVF are designed for millions to billions of vectors (Faiss library paper [R24]). At 23k vectors an exact scan is a ~23.5M multiply-add mat-vec (~94 MB of float32), so approximate indexes only add recall loss. The 8 ms you measured can likely be cut with engineering alone: `np.argpartition` instead of a full sort, a contiguous float32 matrix, or float16 storage **[estimate; engineering suggestion, no published number]**.
- **int8 scalar quantisation** (Shakir, Aarsen and Lee, HF blog 2024 [R23]): 4× smaller index, exact-search speed-up of 2.99× (min), 3.66× (mean) and 4.8× (max) with USearch. Retention was ~97% without rescoring and ~99% with a rescore multiplier of 4–5. Measured on English MTEB retrieval with mxbai-embed-large-v1.
- **Binary quantisation** [R23]: 32× smaller and 15.05× / 24.76× / 45.8× faster (min / mean / max) with Faiss 1.8.0. Retention was 92.53% without rescoring and 96.45% with rescoring. mxbai-embed-large-v1 was trained to be quantisation-friendly; bge-m3 has no such claim, so its retention is **unverified**.
- **Matryoshka truncation** does not apply. bge-m3 is not MRL-trained (no MRL claim found), unlike Qwen3-Embedding, EmbeddingGemma or arctic-m-v2.0 [R3, R5, R21].
- Library availability on Windows (PyPI, checked 2026-11-20 [R49]): `faiss-cpu` 1.16.0 (MIT, win_amd64 wheels), `usearch` 2.26.6 (Apache-2.0, win_amd64 wheels), `hnswlib` 0.8.0 (no Windows wheel, source build). ST provides `quantize_embeddings`, `semantic_search_faiss` and `semantic_search_usearch` [R23]. ST 5.7.0 fixed int8 clipping and FAISS padding bugs [R22c].

### 2.1 Lightweight models from Chinese labs

Not applicable. This stage uses no model, and no Chinese-lab vector-search library offers an advantage over exact NumPy search at 23k vectors. The vector *dimension* comes from the stage-1 model choice: for example, `gte-multilingual-base` gives 768-d (elastic down to 128) and `Qwen3-Embedding-0.6B` gives MRL down to 32 [R7b, R3], but both are weaker on Indonesian than bge-m3 (§1.3).

### 2.2 Stage 2 shortlist

1. **Keep exact NumPy search**, with `argpartition`-based top-40 and float32/float16 contiguous storage. No quality risk.
2. **`usearch` exact int8 search** (or ST `quantize_embeddings(..., precision="int8")` with float rescoring of the top ~160). Evidence: 3.66× mean speed-up and ~99% retention with rescoring (English) [R23].
3. **`faiss-cpu` `IndexBinaryFlat` with float rescoring.** 24.76× mean speed-up and 96.45% retention with rescoring (English, mxbai) [R23]. Validate on bge-m3 first.

Expected user-visible gain: a few milliseconds.

---

## 3. Stage 3 — keyword search (BM25)

Evidence:

- **Language-aware analysis matters more than the engine.**
  - MIRACL baseline (Anserini/Lucene BM25 with language-specific analyzers): Indonesian dev nDCG@10 **0.449**, Recall@100 **0.904**, versus mDPR 0.272 / 0.573 [R26].
  - The bge-m3 paper's BM25, which uses the XLM-R tokenizer, gets only **29.7** on MIRACL id. Its appendix notes that the Lucene analyzer gives a higher average (38.5 vs 31.9) [R1]. These come from different implementations, but they point the same way: BM25 needs proper Indonesian tokenisation and stemming.
  - BM25 is also strong on Mr. TyDi Indonesian (MRR@10 38.2) [R6b].
- **SQLite FTS5** offers only the unicode61, ascii, porter (English-only Porter stemmer) and trigram tokenizers. Other languages need a custom C tokenizer. bm25 uses fixed k1 = 1.2 and b = 0.75 [R27]. The cheapest fix is to **stem text in Python before inserting it into FTS5 and before querying**. The Snowball project includes an **Indonesian** stemmer [R28], and PyStemmer 3.1.0 (MIT/BSD) ships Windows wheels [R49].
- **bm25s** (Lù, arXiv:2407.03618 [R25]): up to 500× faster than the most popular Python BM25 (rank-bm25). Single-thread Xeon QPS examples from the README: scifact 952.92 vs rank-bm25 47.6 and Elasticsearch 20.81; nfcorpus 1196.16 vs 224.66 / 45.84. It accepts any stemmer callable, has an optional Numba backend ("about 2x" on larger datasets), supports memory-mapping, and is MIT-licensed. It is pure Python (no Windows wheel needed). Indonesian stopwords are not documented; supply your own list.
- **tantivy** 0.27.0 (MIT) has a Windows wheel [R49]. Indonesian stemming support in tantivy-py was **not verified**.
- At 4 ms, FTS5 is already fast. Engine changes save at most a few ms. The real gain on offer is **quality** (stemming), which can let you rerank fewer candidates in stage 5.

### 3.1 Lightweight models from Chinese labs

The only relevant items are the learned sparse (lexical) heads of two Chinese-lab embedders. Either could replace BM25 entirely.

- **bge-m3 (BAAI):** MIRACL id sparse alone 48.9; dense+sparse 58.3 vs dense alone 56.1 [R1].
- **gte-multilingual-base (Alibaba):** a similar dense+sparse gain, MIRACL id 50.3 → 56.4 [R7].

Neither paper reports exact-match behaviour on decree numbers and IDs, so keep a lexical channel for those **[uncertain]**.

### 3.2 Stage 3 shortlist

1. **SQLite FTS5 plus Indonesian Snowball stemming done in Python** (PyStemmer `Stemmer.Stemmer("indonesian")`). Keep IDs and decree numbers as unstemmed tokens. Same speed, better recall (supported by MIRACL BM25 0.449 with a language analyzer [R26]).
2. **`bm25s`** (`bm25s[core]` with Numba) with the same stemmer. Sub-millisecond scoring is likely at 23k documents **[estimate]**.
3. **bge-m3 sparse weights** as the lexical channel. They come free with the stage-1 forward pass. MIRACL id dense+sparse 58.3 [R1]. ST 6 support **[uncertain]**.

## 4. Stage 4 — reciprocal rank fusion

No change needed. RRF over 80 ranked IDs costs microseconds and is not a bottleneck.

---

## 5. Stage 5 — reranking (24 candidates → 6)

### 5.1 Indonesian evidence: MIRACL-id nDCG@10 from three independent papers

| Reranker (HF ID) | Params | Licence | jina-reranker-v3 paper [R13] (1st stage jina-emb-v3, top-100) | Querit paper [R16] (1st stage Qwen3-Emb-0.6B) | Lychee/KaLM-Reranker paper [R17] (1st stage KaLM-Emb-V2.5, top-100) |
|---|---|---|---|---|---|
| **BAAI/bge-reranker-v2-m3** (current) | 568M | Apache-2.0 | **58.86** | **56.37** | **61.87** |
| Qwen/Qwen3-Reranker-0.6B | 596M | Apache-2.0 | 49.66 | 49.94 | 61.24 |
| Qwen/Qwen3-Reranker-4B | 4B | Apache-2.0 | 54.90 | 53.78 | — |
| jinaai/jina-reranker-v3 | 597M | CC BY-NC 4.0 | 57.43 | 56.26 | — |
| jinaai/jina-reranker-v2-base-multilingual | 278M | CC BY-NC 4.0 | 56.66 | — | — |
| mixedbread-ai/mxbai-rerank-base-v2 | 494M | Apache-2.0 | 49.48 | — | — |
| Querit/Querit (Querit-Reranker-A0.4B, Baidu/CAS) | MoE, 0.4B active | Apache-2.0 | — | 54.02 | — |
| KaLM-Embedding/KaLM-Reranker-V1-Nano (HIT) | 0.27B activated (card) | Apache-2.0 | — | — | 59.04 |
| First-stage retriever alone | — | — | 47.83 | 51.71 | 49.89 |

**bge-reranker-v2-m3 is the best Indonesian reranker in all three papers.** Qwen3-Reranker-0.6B ranges from 49.7 to 61.2 depending on the paper, so its Indonesian quality is unstable across setups and evaluators. Multilingual averages point the same way:

- On MIRACL, bge-reranker-v2-m3 scores 69.32 vs 57.70 for Qwen3-Reranker-0.6B [R13].
- The Qwen card reports the opposite on its own MMTEB-R setup (Qwen3-Reranker-0.6B 66.36 vs bge 58.36, top-100 from Qwen3-Embedding-0.6B) [R4].

**Smaller cross-encoders without Indonesian MIRACL numbers:**

- `Alibaba-NLP/gte-multilingual-reranker-base` (306M, 12 layers vs 24, Apache-2.0):
  - mGTE paper Table 5: MIRACL avg 68.5 vs bge-reranker-v2-m3 72.6; MKQA Recall@20 67.2 vs 68.7 [R7].
  - Qwen card: MMTEB-R 59.44 vs 58.36 [R4].
  - The card claims 10× faster than decoder-based rerankers [R7b]. No Indonesian-specific number found.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M, Apache-2.0):
  - Trained on mMARCO, a 14-language machine-translated MS MARCO whose language list includes Indonesian (`in`) [R19].
  - jina card: MKQA nDCG@10 (26 languages) 53.37 vs bge-reranker-v2-m3 54.17, but BEIR 45.40 vs 53.65 and MLDR 28.91 vs 59.73 [R15].

### 5.2 Techniques

ST CrossEncoder benchmark [R22b]: RTX 3090 / i7-13700K, models including BAAI/bge-reranker-v2-m3, quality on English NanoBEIR.

| Technique | Speed-up | Quality ratio | Notes |
|---|---|---|---|
| CPU OpenVINO int8 (`export_static_quantized_openvino_model` + `backend="openvino"`) | **3.09×** vs CPU fp32 | 100.58% | Best CPU option in the benchmark; the ST flowchart recommends it when "minor performance degradations" are acceptable. Your i7-14700F is an Intel CPU of the same generation as the test CPU. |
| CPU OpenVINO fp32 | 1.38× | 100.00% | Lossless option. |
| CPU ONNX int8 | 1.36× | 98.77% | — |
| CPU torch fp16/bf16 | 0.17× | — | Do not use half precision on the CPU. |
| GPU fp16 | **2.67×** vs GPU fp32 | 99.78% | Moving from CPU fp32 to GPU fp16 should be far larger than 2.67× **[estimate]**. Use ST ≥ 6.0, which upcasts reranker logits to fp32 before the sigmoid; in bf16 the earlier behaviour "essentially randomized" rankings [R22e]. |
| GPU FlashAttention-2 | ≤ fp16 | — | "Not recommended for Cross Encoders" [R22b]. |
| Fewer candidates (24 → 12–16) | Roughly linear in pairs | Depends on first-stage recall | RAGMark: reranker cost linear in depth [R42]. Over-deep reranking can hurt [R44]. Better stage-3 stemming raises first-stage recall and makes a smaller depth safer. |
| Shorter `max_length` (512 → 384) | Roughly proportional | Truncates long passages (yours average 356, max 450 tokens) | Test; a likely small loss **[uncertain]**. |

**VRAM check for the GPU option [estimate].**

- bge-reranker-v2-m3 in fp16 is about 1.14 GB of weights.
- Qwen3-8B Q4_K_M is 5.2 GB [R35d]. Its f16 KV cache at num_ctx 8192 is 2 × 36 layers × 8 KV heads × 128 × 2 B = 144 KiB/token, about 1.13 GiB (config [R33b]). With `OLLAMA_KV_CACHE_TYPE=q8_0` that drops to about 0.56 GiB [R35a].
- Reranker + LLM + KV is about 7–7.5 GB before compute buffers, which is tight on 8 GB. It is comfortable with a 3–4 GB LLM (§6).

### 5.3 Lightweight models from Chinese labs

- **BAAI:** the v2-m3 card's own table lists `BAAI/bge-reranker-base` and `bge-reranker-large` as **"Chinese and English"** only, so they are confirmed unsuitable. `BAAI/bge-reranker-v2-gemma` (2.5B) and `BAAI/bge-reranker-v2-minicpm-layerwise` (2.7B) are multilingual but 4–5× larger than v2-m3, so they are not faster [R18]. No Indonesian numbers found for either.
- **Qwen/Qwen3-Reranker-0.6B:** same size as v2-m3 (a causal LM scoring yes/no logits); Indonesian results are inconsistent across papers (49.7–61.2, table above). It runs in ST as a CrossEncoder with `LogitScore` (since ST 5.4) [R22f]. It is not faster and not better on Indonesian evidence.
- **Alibaba-NLP/gte-multilingual-reranker-base:** see §5.1; no Indonesian figure found.
- **Querit/Querit (Baidu / CAS):** MIRACL id 54.02 [R16]. The Hub language tags (8 languages) **do not include Indonesian**. Apache-2.0.
- **KaLM-Embedding/KaLM-Reranker-V1-Nano (HIT Shenzhen):** MIRACL id 59.04 vs v2-m3 61.87 in the same paper [R17]. The paper's *theoretical* compute cost is 1.0× vs 24.2× for v2-m3 (no wall-clock numbers). However, the card requires **`sentence-transformers>=5.6,<6` and `transformers>=5.3,<6`**, which is incompatible with your ST 6 / transformers 5.15 stack, plus `trust_remote_code`. Interesting, but blocked.

### 5.4 Stage 5 shortlist

1. **`BAAI/bge-reranker-v2-m3` exported to OpenVINO int8 on the CPU.** 3.09× faster and 100.58% NanoBEIR quality in the ST benchmark [R22b]. It keeps the best Indonesian reranker (top on MIRACL-id in all three papers [R13, R16, R17]) and uses no VRAM.
2. **`BAAI/bge-reranker-v2-m3` in fp16 on the GPU.** Best when the LLM is a 3–4 GB model (§6) or the KV cache is q8_0. GPU fp16 is 2.67× over GPU fp32 with 99.78% quality [R22b]; use ST ≥ 6.0 (fp32 logit upcast).
3. **Either of the above with 12–16 candidates instead of 24.** Linear saving [R42]; quality risk bounded by [R44]. Pair it with Indonesian stemming in stage 3.
4. **`Alibaba-NLP/gte-multilingual-reranker-base`** (306M, half the depth). MIRACL avg −4.1 and MKQA −1.5 vs v2-m3 [R7]. No Indonesian figure, so validate on your own data.
   - Smaller fallback: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M; MKQA −0.8 but BEIR −8.3 [R15]).

Not shortlisted:

- Qwen3-Reranker-0.6B: no faster, and its Indonesian results are inconsistent.
- jina-reranker-v2/v3/v3.5: CC BY-NC 4.0 licence.
- KaLM-Reranker-V1: needs ST < 6.

---

## 6. Stage 6 — answer generation

### 6.1 Indonesian evidence: SEA-HELM leaderboard, Indonesian, models ≤ 12B [R30]

Overall = SEA-HELM Indonesian score. QA = Indonesian extractive QA (ChrF++). IF = SEA-IFEval (Indonesian). Knowledge = Global-MMLU-Lite (Indonesian). Values are means; brackets give 95% CIs.

| Model (Hub) | Params | Overall ID [CI] | QA [CI] | IF | Knowledge | Licence | Q4_K_M size |
|---|---|---|---|---|---|---|---|
| gemma-4-12B-it | 12B | 73.10 [71.6–74.5] | 82.91 [77.2–88.4] | 96.45 | 79.32 | not checked (E2B/E4B are Apache-2.0) | `gemma4:12b` 7.7 GB [R35d], too big for 8 GB with KV |
| **aisingapore/Nemotron-SEA-LION-v5-E4B** | 8.0B total, "E4B" | **72.09 [70.6–73.5]** | 79.24 [72.8–85.2] | 96.42 | 76.09 | MIT | 5.34 GB (`mradermacher/Nemotron-SEA-LION-v5-E4B-GGUF`) |
| aisingapore/Qwen-SEA-LION-v5-9B | 9.65B | 68.82 [67.4–70.2] | 80.69 | 91.47 | 79.11 | MIT | 5.78 GB (`mradermacher/Qwen-SEA-LION-v5-9B-GGUF`) |
| Qwen/Qwen3.5-9B | 9.65B | 68.69 [67.3–69.9] | 81.10 | 94.70 | 77.98 | Apache-2.0 | `qwen3.5:9b` 6.6 GB |
| **aisingapore/Gemma-SEA-LION-v5-E2B** | 5.1B total / 2.3B effective | **63.50 [62.0–64.9]** | 80.37 [74.3–86.1] | 92.49 | 62.53 | MIT | 3.43 GB (`mradermacher/Gemma-SEA-LION-v5-E2B-GGUF`) |
| **Qwen/Qwen3.5-4B** | 4.66B (incl. vision) | **62.73 [61.5–63.9]** | **80.53 [75.2–85.8]** | 89.43 | 71.57 | Apache-2.0 | `qwen3.5:4b` 3.3 GB |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B | 8.0B | 62.56 | 74.99 | 94.99 | 68.67 | OpenMDW-1.1 | `nemotron-3.5-bolt:e4b-q4_K_M` 7.9 GB (incl. vision) |
| **Qwen/Qwen3-8B (current)** | 8.19B | **62.47 [61.1–63.8]** | **75.02 [68.4–81.7]** | 86.96 | 72.27 | Apache-2.0 | `qwen3:8b` 5.2 GB |
| gemma-3-12b-it | 12B | 62.30 | 81.26 | 89.95 | 60.85 | Gemma (checked for 4b) | `gemma3:12b` 8.2 GB |
| gemma-4-E4B-it | 8.0B total | 61.95 | 73.52 | 89.14 | 67.00 | Apache-2.0 | `gemma4:e4b` 6.6 GB (`e4b-it-qat` 6.1 GB) |
| Qwen3-VL-8B-Instruct | 8B | 60.17 | 80.80 | 91.26 | 59.59 | — | — |
| Qwen-SEA-LION-v4-8B-VL | 8B | 60.15 | 81.19 | 93.61 | 59.11 | — | — |
| Gemma-SEA-LION-v3-9B-IT | 9B | 60.05 | 79.46 | 90.08 | 57.37 | — | — |
| Llama-SEA-LION-v3.5-8B-R | 8B | 58.65 | 71.48 | 83.43 | 61.33 | Llama 3.1 | — |
| gemma-4-E2B-it | 5.1B total | 56.99 | 74.75 | 90.23 | 53.30 | Apache-2.0 | `gemma4:e2b` 4.6 GB |
| Gemma-SEA-LION-v4.5-E2B-IT | 5.1B total | 56.87 | 73.23 | 89.17 | 52.95 | MIT | 3.43 GB (official GGUF) |
| Qwen3-VL-4B-Instruct | 4B | 55.03 | 77.12 | 88.86 | 47.50 | — | — |
| Llama-3.1-8B-Instruct | 8B | 49.98 | 76.87 | 78.13 | 40.76 | — | — |
| gemma-3-4b-it | 4.3B | 49.41 | 71.26 | 85.60 | 48.46 | Gemma | `gemma3:4b` 3.4 GB |
| Llama-3.2-3B-Instruct | 3.2B | **31.79** | 65.33 | 62.02 | 11.35 | Llama 3.2 | `llama3.2:3b` 2.0 GB |

**Important caveats.**

- **(a) Thinking mode.** SEA-HELM flags Qwen3, Qwen3.5 and the SEA-LION v5 models as reasoning models. The v5 cards say evaluations used the model-config generation parameters, with thinking on by default [R31]. You run thinking **off**, and non-thinking scores can be lower. For example, Qwen3-4B on Indonesian MMMLU drops from 72.3 (thinking) to 66.5 (non-thinking) [R32]. Rank the candidates in non-thinking mode on your own questions.
- **(b) The QA confidence intervals are wide (about ±6).** Most ≥4B models are statistically tied on QA. The overall score has tighter CIs (about ±1.5).
- **(c) Missing models.** Qwen3-4B, Qwen3-1.7B, Phi-4-mini and Sahabat-AI are **not** on the current leaderboard.

### 6.2 Other Indonesian evidence

- **Qwen3 Technical Report, Table 31 (Indonesian)** [R32], scores for INCLUDE / MMMLU:

  | Model | Thinking | Non-thinking |
  |---|---|---|
  | Qwen3-8B | 72.7 / 77.7 | **78.2 / 69.6** |
  | Qwen3-4B | 70.9 / 72.3 | **67.3 / 66.5** |
  | Qwen3-1.7B | 63.6 / 61.2 | **52.7 / 49.0** |
  | Qwen3-0.6B | 36.4 / 46.6 | 52.7 / 40.0 |

  - Belebele, Austronesian group: Qwen3-4B 73.0 (thinking) / 65.6 (non-thinking); Qwen3-1.7B 58.6 / 53.0; Gemma-3-4B-IT 61.7; Gemma-3-12B-IT 79.3.
  - Conclusion: **Qwen3-1.7B loses a lot of Indonesian ability**. It is not a safe replacement.
- **Qwen3.5-4B card** (multilingual averages, not Indonesian-specific): MMMLU 76.1, INCLUDE 71.0, MMLU-ProX 71.5 [R33].
- **Gemma 4 (Ollama page):** MMMLU E4B 76.6%, E2B 67.4%, vs Gemma 3 27B 70.7% [R35e].
- **Sahabat-AI** (`GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct`, 9.24B, Gemma licence):
  - Its card reports older SEA-HELM/BHASA Bahasa Indonesia 64.154 vs gemma-2-9B 63.394 and Qwen2.5-7B 60.372.
  - On IndoMMLU it scores 62.6% vs 61.4% and 56.0% [R34].
  - It is only compared with 2024-era models and is larger than Qwen3-8B, so it is not a speed candidate. No 2026 small Sahabat-AI release was found. GoTo's latest is a 27B Gemma-3 CPT (Dec 2025).
- **Phi-4-mini** (`microsoft/Phi-4-mini-instruct`, 3.84B, MIT): the Hub language list (23 languages) **does not include Indonesian**. No Indonesian benchmark found. `phi4-mini` is 2.5 GB in Ollama.
- **Llama 3.2:** the official languages are en, de, fr, it, pt, hi, es and th (no Indonesian). SEA-HELM ID is 31.79, so **unsuitable**.
- **Sailor2** (`sail/Sailor2-8B-Chat`, Apache-2.0; `sailor2:8b` 5.2 GB, `sailor2:1b` 1.1 GB in Ollama): not on the current SEA-HELM leaderboard. No 2026 Indonesian score found.

### 6.3 Techniques (independent of model choice)

1. **Cap and shape the output.** Decode time is linear in generated tokens, and decode dominates (§0). Set `num_predict` to roughly 512–768 instead of 1536 and instruct concise, cited answers. This is the single largest, zero-risk lever **[estimate; validate answer completeness]**.
2. **KV-cache quantisation.** `OLLAMA_KV_CACHE_TYPE=q8_0` uses "roughly half the memory of f16" with "very small precision loss". `q4_0` uses about a quarter, with "small-to-medium" loss that may be "more noticeable at higher context sizes". The FAQ notes that models with high GQA ratios (it cites Qwen2) may lose more precision [R35a]. The benefit is VRAM headroom (≈0.56 GiB saved for Qwen3-8B at 8k [estimate]) rather than raw speed.
3. **Flash attention.** Ollama now "uses Flash Attention automatically when the selected backend and devices support it". `OLLAMA_FLASH_ATTENTION=1` forces it on [R35a]. In llama.cpp on a 4060 Ti, FA raised pp512 from 3,395 to 3,803 t/s (+12%) with tg unchanged [R36b].
4. **Avoid reload latency.** Use `keep_alive: -1` (or `OLLAMA_KEEP_ALIVE`), and preload with an empty request [R35a]. `load_duration` shows the cost [R35c].
5. **num_ctx.** Keep it just above the real prompt plus output (measure `prompt_eval_count`). The FAQ: memory scales with `OLLAMA_NUM_PARALLEL × context length`; leave `OLLAMA_NUM_PARALLEL=1` [R35a].
6. **Speculative decoding (lossless).**
   - **Theory.** Leviathan et al. (ICML 2023) report a 2–3× speed-up on T5-XXL with identical outputs [R37].
   - **Most relevant for RAG.** *Inference with Reference* (LLMA) copies spans from the retrieved context and reports "over 2x" speed-up where output overlaps the reference [R38]. Answers that quote decree numbers and table cells fit this case.
     - llama.cpp implements reference-free n-gram variants (`ngram-simple`, `ngram-map-k`, `ngram-mod`, `ngram-cache`) in `llama-server` [R36a].
   - **Draft models for Qwen3-8B.**
     - `AngelSlim/Qwen3-8B_eagle3`: mean accept length 1.99; 257.5 vs 151.8 tok/s (≈1.7×) on an H20 with vLLM [R40]. It is listed as a supported EAGLE-3 draft in llama.cpp [R36a].
     - `z-lab/Qwen3-8B-DFlash-b16` (MIT): claims up to 6.17× for Qwen3-8B in thinking-off mode; the hardware is not stated on the card [R39].
     - DFlash drafts also exist for `Qwen3.5-4B`/`Qwen3.5-9B` (`z-lab/Qwen3.5-4B-DFlash`, `z-lab/Qwen3.5-9B-DFlash`; Hub listing).
     - Qwen3.5 has native MTP heads, and Ollama publishes `qwen3.5:4b-mtp-q4_K_M` (4.1 GB) [R35d].
   - **Caveats.**
     - Draft acceptance on Indonesian text is unknown; these drafts were trained on mostly English and Chinese data **[uncertain]**.
     - In Ollama, automatic MTP speculative decoding is documented for the **Apple MLX** engine [R35b]. The Modelfile documents `draft_num_predict` ("embedded MTP tensors require setting this parameter") [R35f]. Release notes mention llama-server-based draft models (v0.40.7) and bundled DFlash drafts for `nemotron-3.5-bolt:e4b` [R35b]. I found **no explicit confirmation of CUDA/Windows support** in Ollama **[uncertain]**.
     - The verified route on Windows/CUDA is to run `llama-server` directly with `--spec-type ngram-*`, `draft-eagle3` or `draft-dflash` [R36a]. A draft model also costs VRAM.

### 6.4 Lightweight models from Chinese labs (answering LLM)

| Model (lab) | HF ID / Ollama tag | Params | Licence | Indonesian evidence | Verdict |
|---|---|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B (Alibaba) | `Qwen/Qwen3-*`; `qwen3:0.6b` 523 MB, `qwen3:1.7b` 1.4 GB, `qwen3:4b` / `qwen3:4b-instruct` (2507) 2.5 GB, `qwen3:8b` 5.2 GB | 0.6–8.2B | Apache-2.0 | Tech report Table 31 (see §6.2); SEA-HELM ID for 8B 62.47 | 4B is a fallback; 1.7B and 0.6B are too weak in Indonesian. `Qwen/Qwen3-4B-Instruct-2507` (non-thinking only) has no Indonesian benchmark of its own. |
| Qwen3.5 0.8B / 2B / 4B / 9B (Alibaba, Feb 2026) | `qwen3.5:0.8b` 1.2 GB, `:2b` 2.7 GB (`2b-q4_K_M` 1.9 GB), `:4b` 3.3 GB, `:9b` 6.6 GB | 4.66B (4B), 9.65B (9B) | Apache-2.0 | SEA-HELM ID 4B 62.73, 9B 68.69 [R30]; 0.8B and 2B not on the leaderboard | **4B shortlisted.** Hybrid Gated DeltaNet: only 8 of 32 layers use full attention, so the KV cache is ~32 KiB/token vs 144 KiB for Qwen3-8B (card [R33]; [estimate]). Thinking is on by default, so set `think: false` [R35c]. |
| Qwen-SEA-LION-v5-9B (AISG on Qwen3.5-9B) | `aisingapore/Qwen-SEA-LION-v5-9B` | 9.65B | MIT | SEA-HELM ID 68.82 | Quality-up, not speed-up. |
| Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (MoE) | `qwen3.5:35b` 22 GB | 35B / 3B active | Apache-2.0 | SEA-HELM ID 74.87 / 73.65 | Does not fit in VRAM; CPU offload would be slower; 22 GB is tight in 32 GB RAM. |
| DeepSeek-R1 distills (DeepSeek) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `-7B`, `DeepSeek-R1-0528-Qwen3-8B` | 1.8B / 7.6B / 8.2B | MIT | **No Indonesian evidence found**; not on SEA-HELM | Reasoning models: long chains of thought add latency, the opposite of the goal. Not recommended. |
| DeepSeek drafters for Qwen3 (2026-06) | `deepseek-ai/eagle3_qwen3_8b_ttt7`, `dflash_qwen3_8b_block7`, `dspark_qwen3_8b_block7` (also 4B variants) | 1.5–2.3B | **No model card or licence** | — | Interesting for speculative decoding, but undocumented **[uncertain]**. |
| MiniCPM4 / 4.1 (OpenBMB) | `openbmb/MiniCPM4.1-8B`, `MiniCPM4-8B`, `MiniCPM4-0.5B` | 8.19B / 0.43B | Apache-2.0 | Hub languages `zh`, `en` only; **no Indonesian evidence found** | Not recommended. Newer 2026 OpenBMB releases found are OCR and vision models (MiniCPM-OCR, MiniCPM-V-4.7). |
| GLM-4-9B-0414, GLM-Edge (Zhipu / Z.ai) | `zai-org/GLM-4-9B-0414`; `zai-org/glm-edge-1.5b-chat`, `glm-edge-4b-chat` | 9.4B / 1.6B / 4.3B | MIT / GLM-4 licence | Hub languages `zh`, `en` for GLM-4-9B-0414; **no Indonesian evidence found** for GLM-4-9B or GLM-Edge | Not recommended. The only Zhipu model with Indonesian evidence is **GLM-4.7-Flash** (30B-A3B MoE, MIT): SEA-HELM ID 63.09, QA 76.69. Too large for 8 GB VRAM. |
| InternLM3 (Shanghai AI Lab) | `internlm/internlm3-8b-instruct` | 8.8B | Apache-2.0 | Card system prompt mentions English and Chinese; **no Indonesian evidence found** | Not recommended. |
| Hunyuan small (Tencent) | `tencent/Hunyuan-1.8B-Instruct`, `-4B-Instruct`, `-7B-Instruct` (also 0.5B) | 1.8 / 4.2 / 7.5B | Tencent licence (repo LICENSE; not Apache/MIT) **[uncertain details]** | Card benchmarks are zh/en; **no Indonesian evidence found** | Not recommended. Tencent's 2026 Hy4 is 771B. |
| ERNIE 4.5 small (Baidu) | `baidu/ERNIE-4.5-0.3B-PT` | 0.36B | Apache-2.0 | Hub languages `en`, `zh`; **no Indonesian evidence found** | Not recommended. The other ERNIE 4.5/5.5 open models found are 21B+ MoE. |
| Yi-1.5 (01.AI) | `01-ai/Yi-1.5-6B-Chat`, `Yi-1.5-9B-Chat` | 6.1 / 8.8B | Apache-2.0 | **No Indonesian evidence found**; last LLM release 2024 | Not recommended. |

Ollama tags for the DeepSeek, MiniCPM, GLM, InternLM, Hunyuan, ERNIE and Yi models were **not checked**, because none is shortlisted.

### 6.5 Stage 6 shortlist (benchmark locally, thinking off)

1. **`qwen3:8b` (current), tuned.** Cap `num_predict` (≈512–768) and use a concise-answer instruction, `OLLAMA_KV_CACHE_TYPE=q8_0` and `keep_alive=-1`. This is the zero-quality-change baseline. Decode time falls linearly with tokens.
2. **`qwen3.5:4b` (`Qwen/Qwen3.5-4B`, Apache-2.0), `think: false`.**
   - SEA-HELM Indonesian 62.73 vs Qwen3-8B 62.47; Indonesian QA 80.53 vs 75.02 [R30].
   - 3.3 GB vs 5.2 GB, and the KV cache is ~4.5× smaller [estimate]. Expect faster decode and free VRAM for a GPU reranker.
   - Optional: the `qwen3.5:4b-mtp-q4_K_M` tag for MTP speculation, if it works on CUDA **[uncertain]**.
3. **`aisingapore/Gemma-SEA-LION-v5-E2B` (MIT), imported from `mradermacher/Gemma-SEA-LION-v5-E2B-GGUF` Q4_K_M (3.43 GB)** with a Modelfile `FROM <file>.gguf` [R35g]; thinking off.
   - SEA-HELM Indonesian 63.50 (above Qwen3-8B), QA 80.37 [R30]. 2.3B effective parameters [R31].
   - The GGUF is a community conversion, two days old. Compatibility with Ollama's Gemma-4 support is **[uncertain]**. Fallback: the official `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF` (ID 56.87).
4. **`aisingapore/Nemotron-SEA-LION-v5-E4B` (MIT), from `mradermacher/Nemotron-SEA-LION-v5-E4B-GGUF` Q4_K_M (5.34 GB).** The best Indonesian model that fits (SEA-HELM ID 72.09, +9.6 over Qwen3-8B) at about the same file size as `qwen3:8b`.
   - "Low-memory, low time-to-first-token operation" per the card [R31]. Its speed relative to Qwen3-8B on llama.cpp is **not found**: the "E4B" (effective-4B) design suggests faster decode, but this is unverified.
   - Thinking is on by default, so disable it. Treat it as a quality-for-free candidate if it measures at least as fast.

Fallback, not shortlisted: `qwen3:4b-instruct-2507` (2.5 GB, non-thinking by design). Indonesian evidence exists only for the original Qwen3-4B (non-thinking INCLUDE-id 67.3, MMMLU-id 66.5 vs 8B 78.2 / 69.6 [R32]).

---

## 7. Suggested local benchmark protocol (brief)

- **Retrieval (stages 1–5).** Build 100–200 real Indonesian questions with gold passages. Report Recall@24 before reranking, nDCG@6 / Recall@6 after reranking, and median / p95 latency per stage. Compare all quantised variants against the fp32/fp16 originals on the same queries.
- **Generation.** Keep the retrieved context fixed and compare answers with an LLM judge or by hand (faithfulness, correct IDs and decree numbers). Log Ollama's `prompt_eval_duration`, `eval_count` and `eval_duration` to get tokens per second and time to first token [R35c]. Run every LLM with thinking off.
- **VRAM.** Record peak VRAM for each combination (LLM + KV type + GPU reranker or GPU embedder) so the selected set fits in 8 GB with headroom.

---

## 8. Caveats

- Most speed numbers (ST benchmarks, llama.cpp scoreboard, EAGLE-3/DFlash) were measured on hardware or software other than yours. Treat them as ratios, not absolute times.
- ST quality ratios come from English NanoBEIR/STSb, not Indonesian.
- SEA-HELM scores were evaluated with model-default (thinking) settings. Your non-thinking results may rank differently.
- The SEA-LION v5 models and their community GGUFs were 2–9 days old at writing. Check for official GGUF or Ollama releases.
- The Qwen3-Reranker-0.6B Indonesian results conflict across three papers (49.7 / 49.9 / 61.2).
- mE5-large-instruct's MTEB run (46.35) conflicts with its paper (52.1).

---

## References

**Embedding and reranking**

- [R1] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, Z. Liu, "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," arXiv:2402.03216 (BAAI), 2024. https://arxiv.org/abs/2402.03216 (venue not stated on the opened page)
- [R2] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou, "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models," arXiv:2506.05176, 2025. https://arxiv.org/abs/2506.05176
- [R3] Qwen Team, "Qwen/Qwen3-Embedding-0.6B" model card, Hugging Face, 2025. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- [R4] Qwen Team, "Qwen/Qwen3-Reranker-0.6B" model card, Hugging Face, 2025. https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- [R5] H. Schechter Vera, S. Dua, B. Zhang, et al., "EmbeddingGemma: Powerful and Lightweight Text Representations," arXiv:2509.20354, 2025. https://arxiv.org/abs/2509.20354 ; and Google, "google/embeddinggemma-300m" model card. https://huggingface.co/google/embeddinggemma-300m
- [R6] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, F. Wei, "Multilingual E5 Text Embeddings: A Technical Report," arXiv:2402.05672 (Microsoft), 2024. https://arxiv.org/abs/2402.05672
- [R6b] Microsoft, "intfloat/multilingual-e5-small" model card (Mr. TyDi table). https://huggingface.co/intfloat/multilingual-e5-small
- [R7] X. Zhang, Y. Zhang, D. Long, W. Xie, Z. Dai, J. Tang, H. Lin, B. Yang, P. Xie, F. Huang, M. Zhang, W. Li, M. Zhang, "mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval," arXiv:2407.19669 (Alibaba Tongyi Lab), 2024. https://arxiv.org/abs/2407.19669
- [R7b] Alibaba-NLP, "gte-multilingual-base" and "gte-multilingual-reranker-base" model cards. https://huggingface.co/Alibaba-NLP/gte-multilingual-base ; https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base
- [R8] W. Ponwitayarat, R. Ng, J. R. Montalan, T. Aung, J. G. Ngui, Y. Susanto, W. Tjhi, P. Tasawong, E. Cambria, E. Chuangsuwanich, S. Nutanong, P. Limkonchotiwat, "SEA-BED: Southeast Asia Embedding Benchmark," arXiv:2508.12243 (v2), 2025. https://arxiv.org/abs/2508.12243 (a search listing indicates it was later retitled "SEA-BED: How Do Embedding Models Represent Southeast Asian Languages?" and accepted at ACL 2026; that listing was not opened)
- [R9] P. Limkonchotiwat, R. Ng, S. Nutanong, J. G. Ngui, "SEA-Embedding: Open and Reproducible Text Embeddings for Southeast Asia," arXiv:2606.03027, 2026. https://arxiv.org/abs/2606.03027
- [R9b] AI Singapore, "aisingapore/SEA-LION-ModernBERT-Embedding-300M" model card, 2026. https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-300M
- [R10] MTEB maintainers, "embeddings-benchmark/results" (per-task JSON results; MIRACLRetrievalHardNegatives, BelebeleRetrieval and MIRACLReranking files), GitHub, accessed 2026-11-20. https://github.com/embeddings-benchmark/results
- [R11] Microsoft, "microsoft/harrier-oss-v1-0.6b" model card (MIT), 2026. https://huggingface.co/microsoft/harrier-oss-v1-0.6b
- [R12] Jina AI, "jinaai/jina-embeddings-v5-text-small" model card (CC BY-NC 4.0), 2026. https://huggingface.co/jinaai/jina-embeddings-v5-text-small
- [R13] F. Wang, Y. Li, H. Xiao, "jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking," arXiv:2509.25085, 2025. https://arxiv.org/abs/2509.25085 ; and the jinaai/jina-reranker-v3 card. https://huggingface.co/jinaai/jina-reranker-v3
- [R14] Jina AI, "jinaai/jina-reranker-v3.5" model card (CC BY-NC 4.0), 2026. https://huggingface.co/jinaai/jina-reranker-v3.5
- [R15] Jina AI, "jinaai/jina-reranker-v2-base-multilingual" model card (evaluation table including bge-reranker-v2-m3 and mmarco-mMiniLMv2). https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
- [R16] Y. Zhong, J. Yang, W. Huang, Y. Cai, H. Qian, Y. Fan, R. Zhang, L. Su, D. Shi, J. Guo, "Querit-Reranker: Training Compact Multilingual Rerankers via Efficient Label-Free Distribution Adaptation," arXiv:2606.19037, 2026. https://arxiv.org/abs/2606.19037
- [R17] X. Zhao, J. Xu, Z. Dai, X. Zhang, H. Chen, S. Huang, X. Xiong, D. Tang, X. Hu, G. Fu, M. Zhang, B. Hu, "Lychee-Reranker: Fast but Not Late Interaction for Compressed Document Reranking," arXiv:2606.22807 (v5, 16 Oct 2026; the model card cites it as "KaLM-Reranker-V1: …"), 2026. https://arxiv.org/abs/2606.22807 ; and the KaLM-Embedding/KaLM-Reranker-V1-Nano card. https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Nano
- [R18] BAAI, "BAAI/bge-reranker-v2-m3" model card (reranker family table and languages). https://huggingface.co/BAAI/bge-reranker-v2-m3
- [R19] sentence-transformers, "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" model card. https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- [R20] IBM Granite Embedding Team, "ibm-granite/granite-embedding-278m-multilingual" and "-107m-multilingual" model cards, 2024. https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual
- [R21] Snowflake, "Snowflake/snowflake-arctic-embed-m-v2.0" model card, 2024. https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0

**Sentence Transformers, vector search and keyword search**

- [R22a] T. Aarsen et al., "Speeding up Inference — Sentence Transformer," Sentence Transformers documentation (benchmarks re-measured July 2026). https://sbert.net/docs/sentence_transformer/usage/efficiency.html (source: https://github.com/huggingface/sentence-transformers/blob/main/docs/sentence_transformer/usage/efficiency.rst)
- [R22b] T. Aarsen et al., "Speeding up Inference — Cross Encoder," Sentence Transformers documentation. https://sbert.net/docs/cross_encoder/usage/efficiency.html
- [R22c] Hugging Face, sentence-transformers release v5.7.0 (2026-08-06). https://github.com/huggingface/sentence-transformers/releases/tag/v5.7.0
- [R22d] Hugging Face, sentence-transformers release v5.6.1 (2026-07-23). https://github.com/huggingface/sentence-transformers/releases/tag/v5.6.1
- [R22e] Hugging Face, sentence-transformers release v6.0.0 (2026-08-18). https://github.com/huggingface/sentence-transformers/releases/tag/v6.0.0
- [R22f] Hugging Face, sentence-transformers release v5.4.0 (2026-04-09). https://github.com/huggingface/sentence-transformers/releases/tag/v5.4.0
- [R23] A. Shakir, T. Aarsen, S. Lee, "Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval," Hugging Face Blog, 22 Mar 2024. https://huggingface.co/blog/embedding-quantization
- [R24] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, H. Jégou, "The Faiss library," arXiv:2401.08281, 2024. https://arxiv.org/abs/2401.08281
- [R25] X. H. Lù, "BM25S: Orders of magnitude faster lexical search via eager sparse scoring," arXiv:2407.03618, 2024. https://arxiv.org/abs/2407.03618 ; README benchmarks: https://github.com/xhluca/bm25s
- [R26] X. Zhang, N. Thakur, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu, M. Rezagholizadeh, J. Lin, "MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages," *Transactions of the ACL*, vol. 11, pp. 1114–1131, 2023, doi:10.1162/tacl_a_00595. Baseline numbers were read from the arXiv preprint "Making a MIRACL: Multilingual Information Retrieval Across a Continuum of Languages," arXiv:2210.09984, Table 2. https://arxiv.org/abs/2210.09984 ; dataset statistics: https://github.com/project-miracl/miracl
- [R27] SQLite, "SQLite FTS5 Extension." https://www.sqlite.org/fts5.html
- [R28] Snowball, "Stemming algorithms" (includes Indonesian). https://snowballstem.org/algorithms/

**Answering LLMs and Indonesian evaluation**

- [R29] Y. Susanto, A. V. Hulagadri, J. R. Montalan, J. G. Ngui, X. Yong, W. Q. Leong, H. Rengarajan, P. Limkonchotiwat, Y. Mai, W. C. Tjhi, "SEA-HELM: Southeast Asian Holistic Evaluation of Language Models," in *Findings of ACL 2025*, pp. 12308–12336, doi:10.18653/v1/2025.findings-acl.636.
- [R30] AI Singapore, "SEA-HELM Leaderboard — Indonesian Performance," data last updated 20 Nov 2026. https://leaderboard.sea-lion.ai/detailed/ID
- [R31] AI Singapore, model cards for "Qwen-SEA-LION-v5-9B", "Nemotron-SEA-LION-v5-E4B" and "Gemma-SEA-LION-v5-E2B" (last update 2026-11-17) and "Gemma-SEA-LION-v4.5-E2B-IT" (2026). https://huggingface.co/aisingapore/Nemotron-SEA-LION-v5-E4B ; https://huggingface.co/aisingapore/Gemma-SEA-LION-v5-E2B ; https://huggingface.co/aisingapore/Qwen-SEA-LION-v5-9B
- [R32] Qwen Team, "Qwen3 Technical Report," arXiv:2505.09388, 2025 (Table 31 Indonesian; Table 37 Belebele). https://arxiv.org/abs/2505.09388
- [R33] Qwen Team, "Qwen/Qwen3.5-4B" model card, 2026. https://huggingface.co/Qwen/Qwen3.5-4B
- [R33b] Qwen Team, "Qwen/Qwen3-8B" config.json. https://huggingface.co/Qwen/Qwen3-8B
- [R34] GoTo Company and Indosat, "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct" model card (SEA-HELM/BHASA and IndoMMLU tables), 2024. https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct

**Ollama and llama.cpp**

- [R35a] Ollama, "FAQ" (flash attention, K/V cache quantisation, keep_alive, context length). https://github.com/ollama/ollama/blob/main/docs/faq.mdx
- [R35b] Ollama, GitHub release notes v0.15–v0.40.17 (2026). https://github.com/ollama/ollama/releases
- [R35c] Ollama, "API" documentation (`think`, `keep_alive`, timing fields). https://github.com/ollama/ollama/blob/main/docs/api.md
- [R35d] Ollama library tag pages for qwen3, qwen3.5, gemma3, gemma4, llama3.2, phi4-mini, nemotron-3.5-bolt and sailor2 (accessed 2026-11-20). https://ollama.com/library
- [R35e] Ollama, "gemma4" library page (benchmark table). https://ollama.com/library/gemma4
- [R35f] Ollama, "Modelfile" documentation (`draft_num_predict`). https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx
- [R35g] Ollama, "Importing a model" (GGUF import). https://github.com/ollama/ollama/blob/main/docs/import.mdx
- [R36a] ggml-org, "Speculative Decoding," llama.cpp docs. https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- [R36b] GitHub user olegshulyakov and contributors, "Performance of llama.cpp on Nvidia CUDA," llama.cpp GitHub Discussion #15013, started 1 Aug 2025. https://github.com/ggml-org/llama.cpp/discussions/15013

**Speculative decoding**

- [R37] Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding," in *Proc. ICML* (oral), 2023, arXiv:2211.17192. https://arxiv.org/abs/2211.17192
- [R38] N. Yang, T. Ge, L. Wang, B. Jiao, D. Jiang, L. Yang, R. Majumder, F. Wei, "Inference with Reference: Lossless Acceleration of Large Language Models," arXiv:2304.04487, 2023. https://arxiv.org/abs/2304.04487
- [R39] z-lab, "z-lab/Qwen3-8B-DFlash-b16" model card (cites J. Chen, Y. Liang, Z. Liu, "DFlash: Block Diffusion for Flash Speculative Decoding," arXiv:2602.06036, 2026; the arXiv page itself was not opened). https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16
- [R40] Tencent AngelSlim, "AngelSlim/Qwen3-8B_eagle3" model card. https://huggingface.co/AngelSlim/Qwen3-8B_eagle3

**RAG latency**

- [R41] M. Shen, M. Umar, K. Maeng, G. E. Suh, U. Gupta, "Towards Understanding Systems Trade-offs in Retrieval-Augmented Generation Model Inference," arXiv:2412.11854, 2024. https://arxiv.org/abs/2412.11854
- [R42] Z. Feric, A. Taherin, B. Ren, Y. Wang, J. Dy, D. Kaeli, "RAGMark: A Comprehensive Framework for Benchmarking Retrieval-Augmented Generation Systems," arXiv:2609.05760, 2026. https://arxiv.org/abs/2609.05760
- [R43] W. Jiang, S. Subramanian, C. Graves, G. Alonso, A. Yazdanbakhsh, V. Dadu, "RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving," arXiv:2503.14649, 2025. https://arxiv.org/abs/2503.14649
- [R44] M. Jacob, E. Lindgren, M. Zaharia, M. Carbin, O. Khattab, A. Drozdov, "Drowning in Documents: Consequences of Scaling Reranker Inference," ReNeuIR Workshop at SIGIR 2025, arXiv:2411.11767. https://arxiv.org/abs/2411.11767

**Hardware, packages and model metadata**

- [R46] NVIDIA, "GeForce RTX 4060 Family" specifications. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/
- [R48] Hugging Face Hub model API (licences, safetensors parameter counts, creation dates, language tags), accessed 2026-11-20. Used for all "params" and "licence" cells not otherwise cited, including the Chinese-lab models in §1.3, §5.3 and §6.4 (deepseek-ai, openbmb, zai-org, internlm, tencent, baidu, 01-ai, KaLM-Embedding, TencentBAC, Querit and codefuse-ai repositories). https://huggingface.co/api/models/{id}
- [R49] PyPI JSON metadata for faiss-cpu 1.16.0, usearch 2.26.6, bm25s 0.3.16, tantivy 0.27.0, PyStemmer 3.1.0, hnswlib 0.8.0, openvino 2026.4.1 and onnxruntime 1.32.0, accessed 2026-11-20. https://pypi.org/
