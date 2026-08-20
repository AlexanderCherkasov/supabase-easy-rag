# Supabase Easy RAG — Benchmark & Quality Report

Comprehensive quality evaluation against the **Google Research TyDi QA** multilingual benchmark (5,077 queries, 4,488 documents, 11 languages).

---

## 1. Ground-Truth Document Retrieval Metrics (Strict Document ID/Title Match)

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Document Hit Rate @ 1 (Top-1)** | **88.46%** | Ground-truth document ranked #1 |
| **Document Hit Rate @ 3 (Top-3)** | **91.61%** | Ground-truth document in Top-3 chunks |
| **Document Hit Rate @ 5 (Top-5)** | **92.34%** | Ground-truth document in Top-5 chunks |
| **Document Hit Rate @ 10 (Top-10)** | **92.75%** | Ground-truth document in Top-10 chunks |
| **Document MRR** | **0.9012** | Mean Reciprocal Rank on document retrieval |

---

## 2. Fact / Answer Span Extraction Metrics

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Answer Span Recall @ 1** | **84.70%** | Gold answer span contained in Top-1 chunk |
| **Answer Span Recall @ 5** | **91.90%** | Gold answer span contained in Top-5 chunks |
| **Answer Span Recall @ 10** | **92.40%** | Gold answer span contained in Top-10 chunks |

---

## 3. Ingestion & Synchronization Performance

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Chunking Setup** | **400 chars (overlap 50)** | Sub-chunk splitting enabled |
| **Sync Duration** | **6.89s** | Total synchronization time (16 workers) |
| **Throughput** | **651.5 docs/sec** | Incremental Verification (0 changed, 4488 seen) |

---

## 4. Multilingual Breakdown Across 11 Languages

| Language | Queries | Doc Hit @ 1 | Doc Hit @ 5 | Doc MRR | Answer Recall @ 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabic** | 964 | **89.2%** | **92.8%** | **0.908** | **92.0%** |
| **Finnish** | 920 | **90.0%** | **93.0%** | **0.913** | **92.9%** |
| **Russian** | 828 | **91.5%** | **93.8%** | **0.926** | **93.2%** |
| **Telugu** | 668 | **89.1%** | **92.8%** | **0.906** | **93.1%** |
| **Indonesian** | 499 | **89.4%** | **91.8%** | **0.903** | **91.6%** |
| **Swahili** | 475 | **87.4%** | **92.6%** | **0.895** | **91.6%** |
| **English** | 278 | **87.1%** | **89.9%** | **0.883** | **91.0%** |
| **Korean** | 194 | **75.8%** | **85.0%** | **0.795** | **83.0%** |
| **Japanese** | 139 | **82.0%** | **89.2%** | **0.854** | **88.5%** |
| **Bengali** | 111 | **76.6%** | **91.9%** | **0.834** | **90.1%** |
| **Thai** | 1 | **100.0%** | **100.0%** | **1.000** | **100.0%** |
