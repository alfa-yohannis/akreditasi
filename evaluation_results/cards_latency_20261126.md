# Time to the first evidence card — measurements of 26 November 2026

All on the deployment workstation (i7-14700F, 20 cores; RTX 4060 8 GB), the
akreditasi index (24,792 passages), the twenty benchmark questions in
`evaluation/queries.json` unless stated otherwise. Numbers only; no question
or passage text.

## Where the wait goes (submit → first card)

| Part | Median |
|---|---|
| Browser ↔ server round trip through Cloudflare (12 requests) | 158 ms (84–220) |
| Server: request received, JSON parsed, answer job started (`accepted` frame) | 4 ms |
| Question → six passages inside the engine | 3,397 ms |
| — of which reranking the 24 candidates (BGE-reranker-v2-m3, OpenVINO int8) | ~2,800–3,100 ms |
| — encoding the question (PyTorch, CPU) | 89–413 ms (36–44 ms when run alone) |
| — search by wording (FTS5) / by meaning (NumPy) | 47 ms / 2 ms |

Over real HTTP against a private copy of the server (127.0.0.1:8091):
submit → `sources` frame (the cards) 2,970 ms median, max 3,629 ms.

## Faster with the same results (applied)

| Change | Effect |
|---|---|
| Score each distinct passage text once (a median 20.5 of the 24 candidates are distinct; 20 % of reranker work was repeats) | identical scores |
| Feed passages shortest first (less padding per batch) | rerank 3,100 → 2,800 ms; same six in 20 / 20 questions |
| PyTorch question encoder limited to 8 threads, `KMP_BLOCKTIME=0` | encoding 89 → 42 ms; question → six passages 3,292 → 3,078 ms |
| All of the above, over HTTP | submit → cards 2,970 → **2,536 ms**; engine alone 3,397 → **2,435 ms** |

## How many candidates to rerank

Reference: the reranker scoring every fused candidate (median 71 per
question), its top six as the "ideal". Reranker cost 127 ms per candidate.

| Candidates | Rerank time | Ideal six kept | Questions losing 2+ |
|---|---|---|---|
| 6 | 0.8 s | 37.5 % | 18 / 20 |
| 12 | 1.5 s | 55.8 % | 12 / 20 |
| 18 | 2.3 s | 68.3 % | 10 / 20 |
| **24 (kept)** | **3.1 s** | **78.3 %** | 8 / 20 |
| 32 | 4.1 s | 83.3 % | 7 / 20 |
| 40 | 5.1 s | 90.8 % | 4 / 20 |

Points gained per extra second: 26 (6→12), 16 (12→18), 12.5 (18→24), 5 (24→32).

Other settings, same 24 candidates: one batch of 24 3,559 ms; OpenVINO
THROUGHPUT with parallel batches 2,746–4,076 ms (99 % same six); passages cut
at 384 tokens 2,477 ms (92.5 % same six); at 256 tokens 1,565 ms (78.3 %).

## Alternative rerankers

Same 24 candidates per question; agreement with the current reranker's six
and with the ideal six. Cascade: the alternative narrows 24 to k, then the
current reranker picks six of those.

| Reranker | Time for 24 | Current six kept | Ideal six kept |
|---|---|---|---|
| BGE-reranker-v2-m3, OpenVINO int8 (current) | 2.37 s | 100 % | 78.3 % |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, PyTorch | 0.73 s | 51.7 % | 42.5 % |
| — cascade 24 → 8 / 10 / 12 | ~1.5 / 1.7 / 1.9 s | 62.5 / 75.0 / 82.5 % | |
| Alibaba-NLP/gte-multilingual-reranker-base, PyTorch fp32 | 3.23 s | 44.2 % (all six same in 0 / 20) | 31.7 % |
| — cascade 24 → 8 / 10 / 12 | ~4.0 / 4.2 / 4.4 s | 53.3 / 64.2 / 76.7 % | |

Agreement is with the current reranker, not proven correctness; published
Indonesian results (MIRACL-id, three papers, `literature_review.md` §5) rank
BGE-reranker-v2-m3 first. Neither alternative was adopted.
