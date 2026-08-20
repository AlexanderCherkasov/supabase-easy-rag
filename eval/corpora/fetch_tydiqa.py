"""TyDi QA (Typologically Diverse Question Answering) Dataset Fetcher.

Downloads realistic multilingual articles and QA pairs from Google Research's TyDi QA dataset
via HuggingFace streaming rows API:
- Extracts authentic Wikipedia passages (context + title) as documents for the knowledge base.
- Extracts realistic human-authored search queries and gold standard passage spans.
- Formats documents as markdown with headings and metadata.
- Prepares evaluation dataset JSON mapping questions to expected ground-truth document keys and answer spans.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


def clean_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", cleaned)[:50] or "doc"


def fetch_tydiqa_corpus(
    dest_dir: Path,
    limit: int = 500,
    split: str = "train",
) -> Tuple[List[Path], Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = dest_dir / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TyDi QA] Fetching authentic TyDi QA passages & questions (target: {limit} docs) ...")

    qa_items: List[Dict[str, Any]] = []
    seen_contexts = set()
    files: List[Path] = []
    offset = 0
    batch_size = 100

    while len(files) < limit:
        url = f"https://datasets-server.huggingface.co/rows?dataset=google-research-datasets%2Ftydiqa&config=secondary_task&split={split}&offset={offset}&limit={batch_size}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Supabase-Easy-RAG-Eval)"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("rows", [])
                if not rows:
                    break
                for row_wrapper in rows:
                    row = row_wrapper.get("row", {})
                    q_id = row.get("id")
                    title = row.get("title", "Untitled Passage").strip()
                    context = row.get("context", "").strip()
                    question = row.get("question", "").strip()
                    answers = row.get("answers", {})

                    if not context or not question:
                        continue

                    # Context deduplication
                    ctx_hash = hash(context[:200])
                    if ctx_hash in seen_contexts:
                        continue
                    seen_contexts.add(ctx_hash)

                    idx = len(files) + 1
                    safe_title = clean_filename(title)
                    filename = f"tydi_{idx:04d}_{safe_title}.md"
                    file_path = doc_dir / filename

                    # Construct markdown document
                    md_content = f"# {title}\n\n## Overview\n{context}\n"
                    file_path.write_text(md_content, encoding="utf-8")
                    files.append(file_path)

                    # Extract ground truth answers
                    ans_texts = answers.get("text", []) if isinstance(answers, dict) else []

                    qa_items.append({
                        "id": f"tydi_qa_{idx:04d}",
                        "question": question,
                        "expected_document_key": filename,
                        "document_title": title,
                        "gold_answers": ans_texts,
                        "mode": "hybrid",
                    })

                    if len(files) >= limit:
                        break

                offset += batch_size
                print(f"  Fetched {len(files)}/{limit} TyDi QA passages ...")
        except Exception as exc:
            print(f"  Warning during fetch at offset {offset}: {exc}")
            break

    dataset_file = dest_dir / "tydiqa_eval_dataset.json"
    dataset_file.write_text(json.dumps(qa_items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Saved {len(files)} TyDi QA documents to {doc_dir}")
    print(f"✓ Saved {len(qa_items)} ground-truth QA evaluation pairs to {dataset_file}")

    return files, dataset_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/tmp/tydiqa_benchmark", help="Destination folder")
    parser.add_argument("--limit", type=int, default=300, help="Number of documents")
    args = parser.parse_args()
    fetch_tydiqa_corpus(Path(args.dest), limit=args.limit)
