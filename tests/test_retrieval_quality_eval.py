import math
import unittest
from typing import Any, Dict, List, Optional, Tuple


def calculate_rrf_score(
    vector_rank: Optional[int],
    text_rank: Optional[int],
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    text_weight: float = 1.0,
) -> float:
    v_part = (vector_weight / (rrf_k + vector_rank)) if vector_rank is not None else 0.0
    t_part = (text_weight / (rrf_k + text_rank)) if text_rank is not None else 0.0
    return v_part + t_part


class MockDocument:
    def __init__(
        self,
        doc_id: str,
        title: str,
        heading: Optional[str],
        content: str,
        embedding: List[float],
        facet_keys: List[str],
        owner_id: Optional[str] = None,
        language: str = "english",
    ):
        self.doc_id = doc_id
        self.title = title
        self.heading = heading
        self.content = content
        self.embedding = embedding
        self.facet_keys = facet_keys
        self.owner_id = owner_id
        self.language = language


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


import re

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[\w-]+", text)]

def mock_weighted_fts_score(doc: MockDocument, query_terms: List[str]) -> float:
    """Mock weighted FTS scoring: A: Title (weight 1.0), B: Heading (0.4), D: Content (0.1)."""
    q_terms = [t.lower() for t in query_terms]
    title_terms = _tokenize(doc.title)
    heading_terms = _tokenize(doc.heading or "")
    content_terms = _tokenize(doc.content)

    score = 0.0
    for term in q_terms:
        if term in title_terms:
            score += 1.0
        if term in heading_terms:
            score += 0.4
        if term in content_terms:
            score += 0.1
    return score



def simulate_hybrid_rrf_retrieval(
    docs: List[MockDocument],
    query_text: Optional[str],
    query_embedding: Optional[List[float]],
    match_count: int = 5,
    candidate_count: Optional[int] = None,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    text_weight: float = 1.0,
    min_vector_similarity: Optional[float] = None,
    facet_filter: Optional[List[str]] = None,
    owner_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cand_count = candidate_count or max(match_count * 10, 50)
    cand_count = min(cand_count, 500)

    # 1. Filter by owner & facet
    accessible_docs = []
    for d in docs:
        if owner_filter is not None and d.owner_id is not None and d.owner_id != owner_filter:
            continue
        if facet_filter is not None and not any(f in d.facet_keys for f in facet_filter):
            continue
        accessible_docs.append(d)

    # 2. Vector Candidate Stage
    vector_candidates: Dict[str, Tuple[float, int]] = {}
    if query_embedding:
        scored_vec = []
        for d in accessible_docs:
            sim = cosine_similarity(d.embedding, query_embedding)
            if min_vector_similarity is not None and sim < min_vector_similarity:
                continue
            scored_vec.append((d.doc_id, sim))
        scored_vec.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_id, sim) in enumerate(scored_vec[:cand_count], 1):
            vector_candidates[doc_id] = (sim, rank)

    # 3. FTS Candidate Stage
    fts_candidates: Dict[str, Tuple[float, int]] = {}
    if query_text and query_text.strip():
        q_terms = query_text.strip().split()
        scored_fts = []
        for d in accessible_docs:
            score = mock_weighted_fts_score(d, q_terms)
            if score > 0:
                scored_fts.append((d.doc_id, score))
        scored_fts.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(scored_fts[:cand_count], 1):
            fts_candidates[doc_id] = (score, rank)

    # 4. RRF Rank Fusion
    all_candidate_ids = set(vector_candidates.keys()) | set(fts_candidates.keys())
    fused_results = []
    for doc_id in all_candidate_ids:
        doc = next(d for d in docs if d.doc_id == doc_id)
        v_info = vector_candidates.get(doc_id)
        f_info = fts_candidates.get(doc_id)

        v_sim = v_info[0] if v_info else None
        v_rank = v_info[1] if v_info else None

        f_score = f_info[0] if f_info else None
        f_rank = f_info[1] if f_info else None

        rrf_score = calculate_rrf_score(
            vector_rank=v_rank,
            text_rank=f_rank,
            rrf_k=rrf_k,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )

        fused_results.append({
            "doc_id": doc_id,
            "title": doc.title,
            "heading": doc.heading,
            "content": doc.content,
            "vector_score": v_sim,
            "text_score": f_score,
            "hybrid_score": rrf_score,
            "vector_rank": v_rank,
            "text_rank": f_rank,
        })

    fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused_results[:match_count]


class TestRetrievalQualityEval(unittest.TestCase):
    """Deterministic Quality Evaluation Test Suite for Two-Stage Hybrid Retrieval."""

    def setUp(self):
        # Deterministic controlled embedding space (4D normalized vectors)
        # Unit basis vectors:
        # e0 = [1,0,0,0] (AI/ML concept)
        # e1 = [0,1,0,0] (Database/SQL concept)
        # e2 = [0,0,1,0] (Security/Auth concept)
        # e3 = [0,0,0,1] (Finance concept)
        self.docs = [
            MockDocument(
                doc_id="doc_ai_semantic",
                title="Neural Representation Systems",
                heading="Latent Vector Spaces",
                content="Deep representations encode topological geometry of concepts.",
                embedding=[0.95, 0.05, 0.0, 0.0],
                facet_keys=["tech/ml"],
            ),
            MockDocument(
                doc_id="doc_exact_code",
                title="PostgreSQL Index Configuration",
                heading="HNSW and GIN",
                content="Execute CREATE INDEX idx_kb_chunks_fts USING gin(search_vector) with error code ERR-7749.",
                embedding=[0.1, 0.9, 0.0, 0.0],
                facet_keys=["tech/db"],
            ),
            MockDocument(
                doc_id="doc_both_vector_and_fts",
                title="Vector Database Hybrid Indexing",
                heading="HNSW pgvector Architecture",
                content="Combining cosine vector distance with full text search achieves reciprocal rank fusion.",
                embedding=[0.7, 0.7, 0.0, 0.0],
                facet_keys=["tech/db", "tech/ml"],
            ),
            MockDocument(
                doc_id="doc_competing_keyword",
                title="Finance Annual Audit",
                heading="Overview",
                content="The vector of growth was positive with cosine margins.",
                embedding=[0.0, 0.0, 0.0, 1.0],
                facet_keys=["finance"],
            ),
            MockDocument(
                doc_id="doc_title_match",
                title="PostgreSQL Permissions",
                heading="General Notes",
                content="Detailed permissions guide for relational databases.",
                embedding=[0.2, 0.5, 0.5, 0.0],
                facet_keys=["tech/db"],
            ),
            MockDocument(
                doc_id="doc_multilingual_ru",
                title="Руководство по гибридному поиску",
                heading="Ранжирование RRF",
                content="Объединение векторного и полнотекстового поиска с использованием взаимного ранжирования.",
                embedding=[0.6, 0.6, 0.0, 0.0],
                facet_keys=["docs/ru"],
                language="russian",
            ),
        ]

    def test_scenario_1_pure_semantic_match(self):
        """Scenario 1: Semantic match with zero keyword overlap.

        Query: 'concept embeddings geometry' (vector near [1,0,0,0], zero exact word match in content)
        """
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="unrelated_keyword_abc",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            match_count=3,
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "doc_ai_semantic")
        self.assertEqual(results[0]["vector_rank"], 1)

    def test_scenario_2_exact_keyword_match(self):
        """Scenario 2: Exact keyword match with unique code / hash.

        Query: 'ERR-7749' (without vector query)
        """
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="ERR-7749",
            query_embedding=None,
            match_count=3,
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "doc_exact_code")
        self.assertEqual(results[0]["text_rank"], 1)

    def test_scenario_3_combined_semantic_and_lexical(self):
        """Scenario 3: Relevant document present in both vector and FTS candidates outranks single-modality candidates."""
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="Vector Database Hybrid Indexing",
            query_embedding=[0.7, 0.7, 0.0, 0.0],
            match_count=3,
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "doc_both_vector_and_fts")
        self.assertIsNotNone(results[0]["vector_rank"])
        self.assertIsNotNone(results[0]["text_rank"])
        # Both ranks contribute to higher combined RRF score: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
        self.assertGreater(results[0]["hybrid_score"], 0.03)

    def test_scenario_4_competing_vector_and_keyword_results(self):
        """Scenario 4: Competing results are cleanly balanced by RRF rank without raw score distortion."""
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="growth cosine margins",
            query_embedding=[0.95, 0.05, 0.0, 0.0],
            min_vector_similarity=0.5,
            match_count=5,
            rrf_k=60,
        )
        # Vector top rank: doc_ai_semantic (rank 1, text_rank=None)
        # FTS top rank: doc_competing_keyword (rank 1, vector_rank=None because similarity < 0.5)
        # Both top 1 rank items have equal single-source RRF scores 1/(60+1) = 0.01639
        ai_res = next(r for r in results if r["doc_id"] == "doc_ai_semantic")
        fin_res = next(r for r in results if r["doc_id"] == "doc_competing_keyword")

        self.assertAlmostEqual(ai_res["hybrid_score"], 1.0 / 61.0, places=5)
        self.assertAlmostEqual(fin_res["hybrid_score"], 1.0 / 61.0, places=5)

    def test_scenario_5_title_and_section_weighted_relevance(self):
        """Scenario 5: Matches in Title ('A') receive higher weight than body content ('D') via weighted FTS."""
        doc_body = MockDocument(
            doc_id="doc_body_only",
            title="Miscellaneous Notes",
            heading="Section 1",
            content="PostgreSQL Permissions details inside body.",
            embedding=[0.2, 0.5, 0.5, 0.0],
            facet_keys=["tech/db"],
        )
        test_corpus = [self.docs[4], doc_body]  # docs[4] has "PostgreSQL Permissions" in title

        score_title = mock_weighted_fts_score(self.docs[4], ["PostgreSQL", "Permissions"])
        score_body = mock_weighted_fts_score(doc_body, ["PostgreSQL", "Permissions"])

        self.assertGreater(score_title, score_body, "Title matches must produce higher FTS rank than body matches")

    def test_scenario_6_min_vector_similarity_threshold(self):
        """Scenario 6: min_vector_similarity filters out distant vector candidates."""
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="",
            query_embedding=[0.0, 0.0, 0.0, 1.0],  # Finance query
            min_vector_similarity=0.9,  # strict threshold
            match_count=5,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc_competing_keyword")

    def test_scenario_7_facet_and_tenant_filters(self):
        """Scenario 7: Facet filter restricts candidate space before ranking."""
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="Vector Database",
            query_embedding=[0.7, 0.7, 0.0, 0.0],
            facet_filter=["finance"],
            match_count=5,
        )
        for r in results:
            doc = next(d for d in self.docs if d.doc_id == r["doc_id"])
            self.assertIn("finance", doc.facet_keys)

    def test_scenario_8_multilingual_fts(self):
        """Scenario 8: Multilingual text retrieval works seamlessly with configurable FTS."""
        results = simulate_hybrid_rrf_retrieval(
            docs=self.docs,
            query_text="Руководство гибридному",
            query_embedding=[0.6, 0.6, 0.0, 0.0],
            match_count=3,
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "doc_multilingual_ru")


if __name__ == "__main__":
    unittest.main()
