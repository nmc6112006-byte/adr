# ADGR — Adaptive Directed Graph Retrieval (Re-implementation)

Tái hiện đầy đủ framework **ADGR** từ bài báo:

> *"Adaptive Directed Graph Retrieval for Graph-Based Retrieval-Augmented Generation"*
> Ha Nguyen-Tien, Linh Vu-Duy, Minh-Huyen Nguyen-Thi, Toan Le-Ngoc, Dinh-Thai Kim (Applied Intelligence)

Trên tài nguyên **ViMedGraph** — corpus y tế tiếng Việt từ YouMed
(nguồn chính thức: `https://huggingface.co/datasets/vuduylinh150804/vietmed-crawled-corpus`, CC-BY-NC-4.0).

## Kiến trúc (6 blocks — đúng paper)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ BLOCK 1: KG CONSTRUCTION (offline)                                       │
│   Corpus (.txt YouMed) → structure-aware chunking → LLM extraction       │
│   (schema 11 loại thực thể, delimiter <|#|>, temperature=0) → merge      │
│   Mỗi entity/relation lưu source_chunk_ids (mapping π)                   │
├──────────────────────────────────────────────────────────────────────────┤
│ BLOCK 2-5: QUERY-TIME RETRIEVAL                                          │
│   2. Keyword extraction: Kl (low-level), Kh (high-level) bằng LLM        │
│   3. Overlap-aware hybrid anchors:                                       │
│        BM25_kb(κ, V) ⊕ dense top-d (τ_anchor=0.6)                        │
│        s_anchor = min(1, max cos + δ·f(v)), δ=0.1 → Top-k=15             │
│   4. Query-aware semantic edges:                                         │
│        s_sem = cos(z_q, z_e); lọc τ_edge=0.3                             │
│        top-m=10/anchor; s_edge = 0.3·s_anchor + 0.7·s_sem → Top-N_max=50 │
│   5. Multi-source aggregation:                                           │
│        E_glob (Top-n_g=15 qua Kh) + T_s (Top-n_s=10 direct)              │
│        π(·) giữ ⌊r·n/2⌋ chunks nguồn (r=5)                               │
│        Round-robin(Tv, Te, Tg, Ts) + dedup → cross-encoder rerank        │
│        → Top-n_c=10 chunks dưới budget L_max=30.000 tokens               │
├──────────────────────────────────────────────────────────────────────────┤
│ BLOCK 6: GENERATION  a = M(q, C(q)),  temperature=0                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## Cấu trúc repo

```
adgr/
├── configs/default.yaml        # toàn bộ tham số (bảng 2 paper)
├── adgr/                       # package Python
│   ├── config.py               # YAML + env (ADGR_*) + CLI overrides
│   ├── corpus.py               # Document/Chunk + loader sample/HF/dir
│   ├── chunking.py             # structure-aware semantic chunking (Sec 4.2)
│   ├── vitoken.py              # tokenize tiếng Việt (underthesea)
│   ├── kg_schema.py            # 11 loại thực thể
│   ├── kg_prompts.py           # prompt trích xuất (đúng supplementary)
│   ├── kg_extract.py           # parse định dạng <|#|>
│   ├── kg_graph.py             # merge graph + thống kê Bảng 1
│   ├── llm.py                  # LLM backends: mock | openai_compat | vllm_local | zai
│   ├── embeddings.py           # Embedding: hash_mock | st_local
│   ├── rerank.py               # Reranker: none | bge | qwen3 (đúng paper)
│   ├── index_bm25.py           # BM25 entity + chunk
│   ├── index_vec.py            # vector index cosine (NanoVectorDB-style)
│   ├── store.py                # AdgrStore persist/load
│   ├── build_index.py          # 3 stage build
│   ├── retrieval.py            # ⭐ ADGR pipeline Eq.3-21 + trace + ablation
│   ├── generation.py           # Block 6
│   ├── eval_metrics.py         # Hit@K, MRR@K, nDCG@K
│   ├── eval_ragas.py           # RAGAS-style proxy (LLM judge)
│   ├── eval_judge.py           # LLM-as-Judge 4 chiều (nguyên văn supplementary)
│   ├── eval_runner.py          # harness: Bảng 3-7
│   ├── viz.py                  # HTML đồ thị KG + retrieval trace
│   └── cli.py                  # CLI
├── assets/
│   ├── sample_corpus/          # 44 bài YouMed THẬT từ dataset paper
│   └── qa_eval_sample.json     # 16 QA có gold chunk
├── scripts/run_kaggle.py       # one-shot Kaggle
├── kaggle/adgr_kaggle.ipynb    # notebook Kaggle
├── tests/test_e2e_mock.py      # E2E test không cần GPU
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt                 # cốt lõi (chạy được pipeline mock)
pip install sentence-transformers               # + embedding/reranker thật
pip install vllm                                # + LLM local trên GPU (tuỳ chọn)
```

## Chạy nhanh (không cần GPU/API)

```bash
# 1. Test toàn pipeline (mock LLM + hash embedding, ~30 giây)
python tests/test_e2e_mock.py

# 2. Hoặc từng bước bằng CLI
python -m adgr.cli build-corpus --data-dir data
python -m adgr.cli build-kg     --data-dir data
python -m adgr.cli build-index  --data-dir data

# 3. Truy vấn
python -m adgr.cli query "A.t Vildagliptin dùng để điều trị bệnh gì?" --show-trace

# 4. Evaluation (Hit@K, MRR, nDCG, RAGAS-proxy)
python -m adgr.cli eval --out outputs/eval_results.json

# 5. Visualization
python -m adgr.cli viz-kg   --out outputs/kg.html      # đồ thị KG tương tác
python -m adgr.cli viz-trace "Aescin chống chỉ định cho những ai?" --out outputs/trace.html

# 6. Thống kê đồ thị (so sánh Bảng 1)
python -m adgr.cli stats
```

## Chạy trên Kaggle

1. Tạo Notebook, **bật Internet**; (tuỳ chọn) chọn accelerator **GPU T4 x2 / P100**.
2. Upload `adgr.zip` vào `/kaggle/working/` (hoặc dùng notebook đi kèm).
3. Mở `kaggle/adgr_kaggle.ipynb` và chạy tuần tự — hoặc gọn nhất:

```bash
# chế độ kiểm tra pipeline (không GPU):
python scripts/run_kaggle.py --data-dir /kaggle/working/data

# chế độ ĐÚNG PAPER (GPU): Qwen2.5-14B-AWQ + embedding thật + Qwen3-Reranker
python scripts/run_kaggle.py --data-dir /kaggle/working/data_paper \
    --llm vllm_local --llm-model "Qwen/Qwen2.5-14B-Instruct-AWQ" \
    --embed st_local --embed-model "intfloat/multilingual-e5-small" --embed-device cpu \
    --rerank qwen3 --rerank-device cpu \
    --source huggingface --hf-limit 300 --max-chunks-kg 800 --ablation
```

## Backends pluggable

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| `llm_backend` | `mock` | heuristic deterministic — test không cần GPU |
| | `openai_compat` | bất kỳ endpoint OpenAI-compatible (vLLM serve, Ollama, OpenRouter…) |
| | `vllm_local` | vLLM in-process (Kaggle GPU), mặc định `Qwen/Qwen2.5-14B-Instruct-AWQ` như paper |
| | `zai` | z-ai-web-dev-sdk qua node bridge (môi trường tích hợp) |
| `embed_backend` | `hash_mock` | feature hashing 512d — test/CI |
| | `st_local` | sentence-transformers; mặc định `intfloat/multilingual-e5-small`, đổi sang `google/embeddinggemma-300m` để đúng paper |
| `rerank_backend` | `none` | ablation "w/o Cross-Encoder Reranking" |
| | `bge` | `BAAI/bge-reranker-v2-m3` (nhẹ, tốt tiếng Việt) |
| | `qwen3` | `Qwen/Qwen3-Reranker-0.6B` — đúng paper |

Đổi backend bằng CLI: `--set llm_backend=openai_compat --set llm.base_url=http://localhost:8000/v1`
hoặc biến môi trường: `ADGR_LLM_BACKEND=openai_compat`, `ADGR_EMBED_MODEL=google/embeddinggemma-300m`…

## Ablation (Bảng 7 paper)

```bash
python -m adgr.cli eval --ablation          # chạy cả 5 cấu hình
python -m adgr.cli query "..." --mode dense_only      # Dense Retrieval Only
python -m adgr.cli query "..." --mode no_anchor       # w/o Hybrid Anchor Retrieval
python -m adgr.cli query "..." --mode no_edge_filter  # w/o Semantic Edge Filtering
python -m adgr.cli query "..." --mode no_rerank       # w/o Cross-Encoder Reranking
```

## Nạp dữ liệu

| Nguồn | Lệnh |
|---|---|
| 44 bài YouMed thật (đi kèm, từ dataset paper) | `--set corpus.source=sample` |
| Corpus HF đầy đủ (1.932 bài) | `--set corpus.source=huggingface --set corpus.hf_limit=500` |
| Thư mục upload của bạn (.txt/.md, có/không có dòng `# SOURCE_URL:`) | `--set corpus.source=dir --set corpus.upload_dir=/path/to/docs` |

## Lưu ý tái hiện paper

- Toàn bộ tham số Bảng 2 đã mặc định trong `configs/default.yaml` (d=10, τ_anchor=0.6, δ=0.1, k=15, α=0.3/β=0.7, τ_edge=0.3, m=10, N_max=50, n_g=15, n_s=10, r=5, n_c=10, L_max=30k tokens).
- Prompt trích xuất KG bám sát supplementary (delimiter `<|#|>`, `<|COMPLETE|>`, temperature=0, max 4096 tokens).
- Prompt LLM-as-Judge lấy nguyên văn từ supplementary (4 chiều, 0..1, chống thiên vị độ dài).
- Số liệu paper (49.274 entities…) thuộc về corpus đầy đủ 1.932 docs; với sample 44 docs đồ thị sẽ nhỏ hơn nhiều — đó là hành vi đúng.

## Giấy phép dữ liệu

Corpus YouMed trong `assets/sample_corpus/` lấy từ
`vuduylinh150804/vietmed-crawled-corpus` theo license **CC-BY-NC-4.0** —
chỉ dùng cho mục đích nghiên cứu, phi thương mại (chi tiết: `LICENSE-NOTICE.md`).
