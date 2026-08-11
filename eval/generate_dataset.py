#!/usr/bin/env python3
"""
Synthetic Needle-in-a-Haystack (NIAH) Dataset Generator for Supabase Easy RAG.

Generates custom benchmark datasets by placing target needles at controlled relative depths
(0%, 25%, 50%, 75%, 100%) within distractor document context streams.
Optionally calls LLM (OpenAI / Azure) to generate synthetic Q&A pairs from arbitrary Markdown files.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

NEEDLE_DOCS = [
    {
        "filename": "needles/starlight_passkey.md",
        "question": "What is the emergency activation passkey for Project Starlight cluster fra-1-edge?",
        "expected_keywords": ["PASSPHRASE_STARLIGHT_9842", "9443", "fra-1-edge"],
        "needle_fact": "PASSPHRASE_STARLIGHT_9842",
        "mode": "hybrid"
    },
    {
        "filename": "needles/financial_acquisition.md",
        "question": "How much cash did Apex Global pay to acquire Veloce AI in August 2025?",
        "expected_keywords": ["$74.35 million", "Veloce AI", "Apex Global"],
        "needle_fact": "$74.35 million",
        "mode": "hybrid"
    },
    {
        "filename": "needles/medical_dosage.md",
        "question": "What is the maximum permitted daily dosage of Compound-X19 for Syndrome Z-4?",
        "expected_keywords": ["35mg/kg", "8 hours", "Syndrome Z-4"],
        "needle_fact": "35mg/kg",
        "mode": "vector"
    },
    {
        "filename": "needles/db_config_timeout.md",
        "question": "What PostgreSQL statement timeout parameter is required for HNSW reindexing operations?",
        "expected_keywords": ["POSTGRES_STATEMENT_TIMEOUT_MS=14200", "HNSW"],
        "needle_fact": "POSTGRES_STATEMENT_TIMEOUT_MS=14200",
        "mode": "fts"
    },
    {
        "filename": "needles/rls_audit_hash.md",
        "question": "What is the compliance verification hash for RLS audit events?",
        "expected_keywords": ["RLS_AUDIT_HASH_7719B", "Row Level Security"],
        "needle_fact": "RLS_AUDIT_HASH_7719B",
        "mode": "hybrid"
    }
]


def load_haystack_files(base_dir: Path) -> List[Path]:
    """Return all markdown files in haystack directory."""
    if not base_dir.exists():
        return []
    return sorted(list(base_dir.rglob("*.md")))


def generate_depth_variations(
    haystack_dir: Path,
    output_dir: Path,
    depths: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]
) -> List[Dict[str, Any]]:
    """Generate synthetic documents with needles embedded at precise depth ratios."""
    distractor_files = list((haystack_dir / "distractors").rglob("*.md")) if (haystack_dir / "distractors").exists() else []
    distractor_content = []
    for df in distractor_files:
        distractor_content.append(df.read_text(encoding="utf-8"))

    combined_distractors = "\n\n".join(distractor_content) if distractor_content else "Filler text content for haystack generation.\n"

    dataset_items = []
    synth_docs_dir = output_dir / "synthetic_docs"
    synth_docs_dir.mkdir(parents=True, exist_ok=True)

    for idx, needle in enumerate(NEEDLE_DOCS, 1):
        needle_path = haystack_dir / needle["filename"]
        needle_text = needle_path.read_text(encoding="utf-8") if needle_path.exists() else needle["needle_fact"]

        for depth in depths:
            depth_pct = int(depth * 100)
            split_idx = int(len(combined_distractors) * depth)
            before_text = combined_distractors[:split_idx]
            after_text = combined_distractors[split_idx:]

            synthetic_doc_content = f"{before_text}\n\n<!-- NEEDLE START -->\n{needle_text}\n<!-- NEEDLE END -->\n\n{after_text}"
            doc_key = f"synthetic_docs/needle_{idx}_depth_{depth_pct}.md"
            (output_dir / doc_key).write_text(synthetic_doc_content, encoding="utf-8")

            dataset_items.append({
                "id": f"synth_niah_{idx}_depth_{depth_pct}",
                "question": needle["question"],
                "expected_document_key": doc_key,
                "expected_keywords": needle["expected_keywords"],
                "needle_fact": needle["needle_fact"],
                "needle_depth_percent": depth_pct,
                "mode": needle["mode"]
            })

    return dataset_items


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic NIAH Datasets for RAG Eval")
    parser.add_argument("--haystack-dir", default="eval/data/haystack", help="Base haystack docs directory")
    parser.add_argument("--output-json", default="eval/dataset_niah_synthetic.json", help="Output JSON dataset path")
    parser.add_argument("--output-docs-dir", default="eval/data", help="Output directory for generated synthetic docs")
    args = parser.parse_args()

    haystack_path = Path(args.haystack_dir)
    output_docs = Path(args.output_docs_dir)

    print(f"Generating synthetic depth-varied NIAH dataset from {haystack_path}...")
    dataset_items = generate_depth_variations(haystack_path, output_docs)

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(dataset_items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully generated {len(dataset_items)} evaluation items -> {out_json}")


if __name__ == "__main__":
    main()
