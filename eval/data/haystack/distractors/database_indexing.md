# Relational Database Indexing Fundamentals

## B-Tree vs Hash Indexing
B-Tree indexes are the default index structure in PostgreSQL and MySQL, designed for equality (`=`) and range operators (`<`, `<=`, `>`, `>=`). Hash indexes only support basic equality checks but offer O(1) performance lookup times.

## Full-Text Search Indexing (GIN & GiST)
PostgreSQL supports advanced lexical analysis using Generalized Inverted Indexes (GIN). Documents are parsed into tokens, stemmed using language-specific dictionaries, and indexed into tsvector format.

## Write-Ahead Logging (WAL) & Checkpoints
Database transactions are written sequentially to the Write-Ahead Log before being flushed to data pages on disk. Checkpoints periodically synchronize dirty buffers from shared_buffers to storage.

## Query Optimizer Statistics
`ANALYZE` updates table statistics stored in `pg_statistic` to help the planner choose optimal scan types (Seq Scan, Index Scan, Bitmap Index Scan).
