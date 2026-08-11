from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Callable

# Portable Eval Framework for Supabase Easy RAG
# - No hard dep on Supabase: accepts any retriever callable
# - Metrics: Recall@k, HitRate, MRR, KeywordRecall
# - Optional LLM judge via Azure gpt-5-nano / mini

@dataclass
class EvalItem:
    id: str
    question: str
    expected_document_key: str | None = None
    expected_keywords: List[str] | None = None
    mode: str = "hybrid"

@dataclass
class EvalResult:
    id: str
    question: str
    hit: bool
    recall_at_k: float
    mrr: float
    keyword_recall: float
    latency_ms: int
    top_k_keys: List[str]
    top_score: float | None

@dataclass
class EvalReport:
    dataset: str
    model: str
    chunk_size: int
    chunk_overlap: int
    total: int
    hit_rate: float
    avg_recall: float
    mrr: float
    avg_keyword_recall: float
    avg_latency_ms: float
    results: List[EvalResult]


def keyword_recall(keywords: List[str], text: str) -> float:
    if not keywords:
        return 1.0
    t = text.lower()
    hits = sum(1 for k in keywords if k.lower() in t)
    return hits / len(keywords)


def evaluate(
    dataset_path: str | Path = "eval/dataset.json",
    retriever: Optional[Callable[[str, str, int], List]] = None,
    k: int = 5,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    model: str = "text-embedding-3-small",
    use_llm_judge: bool = False,
) -> EvalReport:
    """Run eval. retriever(question, mode, k) -> list[SearchResult|dict with document_key/chunk_text]."""
    dataset_path = Path(dataset_path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    valid_fields = set(EvalItem.__dataclass_fields__.keys())
    items = [EvalItem(**{k: v for k, v in r.items() if k in valid_fields}) for r in raw]

    # Lazy import real client if no retriever provided — explicit connector in example style
    if retriever is None:
        from supabase_easy_rag import EasyRagClient
        from supabase_easy_rag.config import EasyRagConfig, get_config
        from supabase_easy_rag.providers.azure import AzureEmbeddingProvider

        get_config.cache_clear()
        cfg = EasyRagConfig.from_env()
        emb = AzureEmbeddingProvider(api_key=cfg.embedding.api_key, endpoint=cfg.embedding.endpoint or "", model=cfg.embedding.model)
        client = EasyRagClient(embedding_provider=emb)

        def _retriever(q, mode, kk):
            if mode == "vector":
                return client.search_vector(q, match_count=kk)
            elif mode == "fts":
                return client.search_fts(q, match_count=kk)
            else:
                return client.search_hybrid(q, match_count=kk)

        retriever = _retriever

    results: List[EvalResult] = []
    for item in items:
        t0 = time.time()
        hits = retriever(item.question, item.mode, k)
        latency = int((time.time() - t0) * 1000)
        # Normalize keys
        keys = []
        texts = []
        scores = []
        for h in hits:
            if isinstance(h, dict):
                keys.append(h.get("document_key") or h.get("document_id") or "")
                texts.append(h.get("chunk_text") or h.get("content") or "")
                scores.append(h.get("hybrid_score") or h.get("vector_score") or 0)
            else:
                # SearchResult
                # chunk_text contains metadata with facet_path but expected is document_key
                # we approximate via document_title / facet_path; for eval, check keywords instead
                keys.append(getattr(h, "document_id", "") or getattr(h, "facet_path", "") or getattr(h, "document_title", ""))
                texts.append(getattr(h, "chunk_text", ""))
                scores.append(getattr(h, "hybrid_score", None) or getattr(h, "vector_score", None) or getattr(h, "text_score", 0) or 0)
        # Hit if expected key in top k OR keyword recall >0.5
        hit = False
        if item.expected_document_key:
            hit = any(item.expected_document_key.lower() in (kk or "").lower() for kk in keys)
        if not hit and item.expected_keywords:
            # keyword in any retrieved text
            all_text = " ".join(texts).lower()
            hit = keyword_recall(item.expected_keywords, all_text) >= 0.5

        recall = 1.0 if hit else 0.0
        # MRR: 1/rank of first hit
        mrr = 0.0
        for rank, kk in enumerate(keys, 1):
            if item.expected_document_key and item.expected_document_key.lower() in (kk or "").lower():
                mrr = 1.0 / rank
                break
            if item.expected_keywords and keyword_recall(item.expected_keywords, texts[rank-1] if rank-1 < len(texts) else "") >= 0.5:
                mrr = 1.0 / rank
                break

        kre = keyword_recall(item.expected_keywords or [], " ".join(texts))
        results.append(EvalResult(
            id=item.id, question=item.question, hit=hit,
            recall_at_k=recall, mrr=mrr, keyword_recall=kre,
            latency_ms=latency, top_k_keys=keys[:k], top_score=scores[0] if scores else None
        ))

    hit_rate = sum(r.hit for r in results) / len(results) if results else 0
    avg_recall = sum(r.recall_at_k for r in results) / len(results) if results else 0
    mrr_avg = sum(r.mrr for r in results) / len(results) if results else 0
    avg_kw = sum(r.keyword_recall for r in results) / len(results) if results else 0
    avg_lat = sum(r.latency_ms for r in results) / len(results) if results else 0

    report = EvalReport(
        dataset=str(dataset_path), model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        total=len(results), hit_rate=hit_rate, avg_recall=avg_recall, mrr=mrr_avg,
        avg_keyword_recall=avg_kw, avg_latency_ms=avg_lat, results=results
    )

    # Optional LLM judge (Azure nano/mini) - lightweight faithfulness check
    if use_llm_judge:
        try:
            from eval.providers.azure_llm import judge_faithfulness
            for r in report.results:
                # find original item
                item = next(i for i in items if i.id == r.id)
                hits = retriever(item.question, item.mode, k)
                context = "\n\n".join([getattr(h, "chunk_text", "") if not isinstance(h, dict) else h.get("chunk_text","") for h in hits[:3]])
                r.keyword_recall = judge_faithfulness(item.question, context)  # reuse field for demo
        except Exception as e:
            print(f"LLM judge skipped: {e}")

    return report


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Supabase Easy RAG Eval")
    ap.add_argument("--dataset", default="eval/dataset.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", default="eval/output/report.json")
    ap.add_argument("--llm-judge", action="store_true", help="Use Azure gpt-5-nano/mini as judge")
    ap.add_argument("--mock", action="store_true", help="Run with mock retriever (no Supabase needed)")
    args = ap.parse_args()

    if args.mock:
        # Dynamic mock retriever scanning local haystack & content files
        search_dirs = [Path("eval/data/haystack"), Path("content"), Path(".")]
        local_files = {}
        for sdir in search_dirs:
            if sdir.exists():
                for f in sdir.rglob("*.md"):
                    rel = str(f)
                    local_files[rel] = f.read_text(encoding="utf-8")

        def mock_retriever(q, mode, kk):
            q_lower = q.lower()
            terms = [t for t in q_lower.split() if len(t) > 2]
            scored = []
            for path, text in local_files.items():
                t_lower = text.lower()
                sc = sum(1.0 for term in terms if term in t_lower)
                if sc > 0:
                    scored.append({"document_key": path, "chunk_text": text[:600], "hybrid_score": sc})
            scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
            if not scored:
                scored = [{"document_key": "guides/rag-permissions.md", "chunk_text": "owner_id auth.uid() RLS document_owners hybrid search 0.7 vector", "hybrid_score": 0.9}]
            return scored[:kk]
        retriever = mock_retriever
    else:
        retriever = None

    report = evaluate(dataset_path=args.dataset, retriever=retriever, k=args.k, use_llm_judge=args.llm_judge)
    out = Path(args.output)
    out.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Eval done: hit_rate={report.hit_rate:.2f} mrr={report.mrr:.2f} kw={report.avg_keyword_recall:.2f} latency={report.avg_latency_ms:.0f}ms")
    print(f"Report -> {out}")
    for r in report.results:
        status = "✓" if r.hit else "✗"
        print(f" {status} {r.id}: hit={r.hit} mrr={r.mrr:.2f} kw={r.keyword_recall:.2f} {r.top_score}")

if __name__ == "__main__":
    main()
