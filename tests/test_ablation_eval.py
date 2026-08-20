"""Deterministic Unit & Acceptance Tests for Vector / FTS / Hybrid Retrieval Ablation."""

from __future__ import annotations

import unittest
from eval.ablation_study import (
    AblationDoc,
    AblationQuery,
    get_ablation_benchmark_dataset,
    run_ablation_study,
    simulate_retrieval,
)


class TestAblationRetrievalSuite(unittest.TestCase):
    """Verifies that Hybrid RRF is resilient across all query archetypes and beats single modalities."""

    def setUp(self):
        self.corpus, self.queries = get_ablation_benchmark_dataset()

    def test_pure_semantic_query_vector_vs_fts(self):
        """Pure semantic queries with zero keyword match succeed in Vector and fail in FTS."""
        sem_query = next(q for q in self.queries if q.query_id == "q_sem_1")

        res_vec = simulate_retrieval(self.corpus, sem_query, mode="vector")
        res_fts = simulate_retrieval(self.corpus, sem_query, mode="fts")
        res_hybrid = simulate_retrieval(self.corpus, sem_query, mode="hybrid")

        # Vector ranks target doc #1
        self.assertEqual(res_vec[0]["doc_id"], sem_query.target_doc_id)
        # FTS has no lexical overlap -> 0 results
        self.assertEqual(len(res_fts), 0)
        # Hybrid still cleanly ranks target doc #1 via RRF fusion
        self.assertEqual(res_hybrid[0]["doc_id"], sem_query.target_doc_id)

    def test_exact_identifier_query_fts_vs_vector(self):
        """Exact error codes / hashes succeed in FTS with rank 1."""
        code_query = next(q for q in self.queries if q.query_id == "q_code_1")

        res_fts = simulate_retrieval(self.corpus, code_query, mode="fts")
        res_hybrid = simulate_retrieval(self.corpus, code_query, mode="hybrid")

        self.assertEqual(res_fts[0]["doc_id"], code_query.target_doc_id)
        self.assertEqual(res_hybrid[0]["doc_id"], code_query.target_doc_id)

    def test_overall_ablation_study_execution(self):
        """Executes the full ablation benchmark and asserts valid report generation."""
        metrics = run_ablation_study()
        self.assertIn("vector", metrics)
        self.assertIn("fts", metrics)
        self.assertIn("hybrid", metrics)

        # Hybrid MRR should be >= single modalities
        h_mrr = metrics["hybrid"]["mrr"]
        v_mrr = metrics["vector"]["mrr"]
        f_mrr = metrics["fts"]["mrr"]

        self.assertGreaterEqual(h_mrr, v_mrr)
        self.assertGreaterEqual(h_mrr, f_mrr)


if __name__ == "__main__":
    unittest.main()
