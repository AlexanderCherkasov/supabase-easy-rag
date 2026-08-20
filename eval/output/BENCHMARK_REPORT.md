# Supabase Easy RAG — Official Benchmark & Quality Report

Comprehensive quality evaluation and multilingual benchmark report for `supabase-easy-rag` (v0.2.0) evaluated against the **Google Research TyDi QA** benchmark across 11 typologically diverse languages.

---

## 1. Executive Summary

* **Database Engine**: PostgreSQL 17.6 on Supabase Cloud.
* **Corpus Size**: **4,488 authentic documents** across 11 languages.
* **Evaluation Queries**: **5,077 real human questions** with gold-standard answer spans.
* **Ingestion Throughput**: **732.2 documents/second** (4,488 documents synchronized in 6.13 seconds with 8 parallel workers).
* **Buffer Cache Hit Ratio**: **100.0%** (all vector and lexical indices served directly from RAM Shared Buffers).

### Global Information Retrieval (IR) Quality:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Hit Rate @ 1 (Top-1 Accuracy)** | **89.64%** | Ground-truth relevant document is ranked #1 in 89.6% of queries |
| **Hit Rate @ 3 (Top-3 Accuracy)** | **92.71%** | Top-3 retrieval recall |
| **Hit Rate @ 5 (Top-5 Accuracy)** | **93.52%** | Standard LLM context budget retrieval accuracy |
| **Hit Rate @ 10 (Top-10 Accuracy)** | **93.91%** | Broad candidate recall |
| **MRR (Mean Reciprocal Rank)** | **0.9128** | Average reciprocal rank across all 5,077 queries |
| **Answer Span Recall @ 5** | **92.77%** | Exact fact-answer span is contained in the top-5 retrieved chunks |

---

## 2. Multilingual Accuracy Breakdown (11 Languages)

| Language | Language Family | Queries | Hit Rate @ 1 | Hit Rate @ 5 | MRR | Answer Recall @ 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabic** | Afroasiatic (Semitic) | 964 | **90.0%** | **93.8%** | **0.916** | **92.8%** |
| **Finnish** | Uralic (Agglutinative) | 920 | **91.6%** | **94.6%** | **0.929** | **94.0%** |
| **Russian** | Indo-European (Slavic) | 828 | **91.8%** | **94.0%** | **0.928** | **93.2%** |
| **Telugu** | Dravidian | 668 | **90.7%** | **94.8%** | **0.924** | **94.6%** |
| **Indonesian** | Austronesian | 499 | **92.2%** | **94.2%** | **0.929** | **93.8%** |
| **Swahili** | Niger-Congo (Bantu) | 475 | **89.5%** | **94.1%** | **0.913** | **92.8%** |
| **English** | Germanic | 278 | **86.7%** | **90.6%** | **0.883** | **90.3%** |
| **Korean** | Koreanic | 194 | **75.3%** | **85.0%** | **0.793** | **83.0%** |
| **Japanese** | Japonic | 139 | **82.7%** | **89.9%** | **0.863** | **89.2%** |
| **Bengali** | Indo-Aryan | 111 | **77.5%** | **92.8%** | **0.843** | **90.1%** |
| **Thai** | Kra-Dai | 1 | **100.0%** | **100.0%** | **1.000** | **100.0%** |

---

## 3. Search Architecture & Algorithmic Complexity

1. **Two-Stage Hybrid Search (RRF)**:
   * **Stage 1 (Vector Scan)**: $\mathcal{O}(\log N)$ HNSW index scan (`vector_cosine_ops`, $M=16$, $ef\_construction=64$).
   * **Stage 2 (Lexical Scan)**: $\mathcal{O}(1)$ GIN index scan with Top-200 candidate pre-filtering and weighted term scoring (`A`: Title, `B`: Heading, `D`: Content).
   * **Fusion**: In-memory Reciprocal Rank Fusion ($k=60$) combining ranks without full table scan or score normalization distortions.
2. **Context Expansion**:
   * Multi-scale retrieval decoupling: search by dense, high-accuracy chunks (400–600 tokens), with zero-copy expansion to full parent sections (up to 3,000+ tokens) for generation.
