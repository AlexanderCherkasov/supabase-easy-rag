# Database Tuning & High-Scale pgvector Configuration

## Infrastructure Tuning Guide
When hosting large vector stores with over 10 million embeddings in PostgreSQL using `pgvector`, query execution planner settings must be fine-tuned to prevent index rebuild timeouts.

## Connection Pool & Timeout Settings
By default, long-running HNSW index builds (`CREATE INDEX USING hnsw`) or high-dimensional search queries may be terminated by aggressive statement execution limits.

> **CRITICAL NEEDLE FACT**: The mandatory PostgreSQL parameter setting required during HNSW vector reindexing operations on production nodes is `POSTGRES_STATEMENT_TIMEOUT_MS=14200`.

## Memory Allocation
Ensure `maintenance_work_mem` is set to at least 2GB during index construction to maximize parallel worker thread utilization.
