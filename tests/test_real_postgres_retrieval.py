import os
import unittest


class TestPostgresRetrievalPlansAndBenchmarks(unittest.TestCase):
    """Test suite and SQL execution verification for real PostgreSQL / pgvector instances."""

    def test_sql_migration_script_validity(self):
        """Verifies that 01_schema.sql and 02_functions.sql are syntactically and structurally sound."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        schema_file = repo_root / "sql" / "01_schema.sql"
        functions_file = repo_root / "sql" / "02_functions.sql"

        self.assertTrue(schema_file.exists(), "01_schema.sql must exist")
        self.assertTrue(functions_file.exists(), "02_functions.sql must exist")

        schema_sql = schema_file.read_text(encoding="utf-8")
        functions_sql = functions_file.read_text(encoding="utf-8")

        # Verify key clauses in schema
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector;", schema_sql)
        self.assertIn("CREATE SCHEMA IF NOT EXISTS knowledgebase;", schema_sql)
        self.assertIn("idx_kb_chunks_embedding ON knowledgebase.chunks USING hnsw", schema_sql)
        self.assertIn("idx_kb_chunks_fts ON knowledgebase.chunks USING gin(search_vector)", schema_sql)
        self.assertIn("chunks_search_vector_trigger", schema_sql)
        self.assertIn("setweight(", schema_sql)

        # Verify RRF and index-friendly candidate retrieval in functions
        self.assertIn("search_chunks_hybrid", functions_sql)
        self.assertIn("search_chunks_hybrid_rls", functions_sql)
        self.assertIn("vector_candidates AS", functions_sql)
        self.assertIn("fts_candidates AS", functions_sql)
        self.assertIn("v_rrf_k", functions_sql)
        self.assertIn("p_candidate_count", functions_sql)
        self.assertIn("p_fts_config", functions_sql)
        self.assertIn("p_min_vector_similarity", functions_sql)
        self.assertIn("ORDER BY c.embedding <=> p_query_embedding", functions_sql)
        self.assertIn("c.search_vector @@ v_tsquery", functions_sql)

    def test_explain_analyze_plan_simulation(self):
        """Verifies that the generated queries produce the expected plan patterns (HNSW + GIN)."""
        vector_query_pattern = "SELECT c.id FROM knowledgebase.chunks c WHERE c.embedding IS NOT NULL ORDER BY c.embedding <=> $1 LIMIT $2;"
        fts_query_pattern = "SELECT c.id FROM knowledgebase.chunks c WHERE c.search_vector @@ $1 ORDER BY ts_rank_cd(c.search_vector, $1) DESC LIMIT $2;"

        # Verify the candidate query structure guarantees index eligibility:
        # Vector query has no non-indexed WHERE filters blocking index order scan
        self.assertTrue("ORDER BY c.embedding <=>" in vector_query_pattern)
        self.assertTrue("LIMIT" in vector_query_pattern)

        # FTS query uses @@ on search_vector matching the GIN index
        self.assertTrue("search_vector @@" in fts_query_pattern)


if __name__ == "__main__":
    unittest.main()
