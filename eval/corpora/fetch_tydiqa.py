"""TyDi QA (Typologically Diverse Question Answering) Dataset Fetcher.

Downloads realistic multilingual articles and QA pairs from Google Research's TyDi QA dataset
via HuggingFace streaming rows API:
- Extracts authentic Wikipedia passages (context + title) as documents for the knowledge base.
- Extracts realistic human-authored search queries and gold standard passage spans.
- Formats documents as markdown with language and FTS metadata.
- Prepares evaluation dataset JSON mapping questions to expected ground-truth document keys, titles, language, and answer spans.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def clean_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", cleaned)[:50] or "doc"


def extract_language_from_id(row_id: str) -> str:
    if not row_id:
        return "english"
    prefix = row_id.split("-")[0].lower()
    known_languages = {
        "arabic", "bengali", "english", "finnish", "indonesian",
        "japanese", "korean", "russian", "swahili", "telugu", "thai",
    }
    return prefix if prefix in known_languages else "english"


def get_fts_config_for_language(language: str) -> str:
    supported_configs = {"english", "russian", "arabic", "finnish", "indonesian", "swahili"}
    return language if language in supported_configs else "simple"


def fetch_tydiqa_corpus(
    dest_dir: Path,
    limit: int | None = None,
    split: str = "validation",
) -> Tuple[List[Path], Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = dest_dir / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TyDi QA] Fetching authentic TyDi QA passages & questions (split={split}, limit={limit or 'all'}) ...")

    qa_items: List[Dict[str, Any]] = []
    context_to_filename: Dict[str, Tuple[str, str, str]] = {}  # ctx_hash -> (filename, title, language)
    files: List[Path] = []
    offset = 0
    batch_size = 100

    while True:
        url = f"https://datasets-server.huggingface.co/rows?dataset=google-research-datasets%2Ftydiqa&config=secondary_task&split={split}&offset={offset}&limit={batch_size}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Supabase-Easy-RAG-Eval)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("rows", [])
                if not rows:
                    break
                for row_wrapper in rows:
                    row = row_wrapper.get("row", {})
                    row_id = str(row.get("id", ""))
                    title = row.get("title", "Untitled Passage").strip()
                    context = row.get("context", "").strip()
                    question = row.get("question", "").strip()
                    answers = row.get("answers", {})

                    if not context or not question:
                        continue

                    lang = extract_language_from_id(row_id)
                    fts_cfg = get_fts_config_for_language(lang)

                    # Deduplicate contexts so multiple QA pairs can map to the same document
                    ctx_key = f"{lang}:{context.strip()}"
                    ctx_hash = hash(ctx_key)

                    if ctx_hash in context_to_filename:
                        filename, doc_title, doc_lang = context_to_filename[ctx_hash]
                    else:
                        idx = len(files) + 1
                        filename = f"tydi_{idx:05d}.md"
                        file_path = doc_dir / filename

                        # Construct markdown document with language & FTS metadata
                        md_content = (
                            f"# {title}\n\n"
                            f"## Metadata\n"
                            f"- **Language**: {lang}\n"
                            f"- **FTS Config**: {fts_cfg}\n\n"
                            f"## Overview\n"
                            f"{context}\n"
                        )
                        file_path.write_text(md_content, encoding="utf-8")
                        files.append(file_path)
                        context_to_filename[ctx_hash] = (filename, title, lang)

                    # Extract gold answer spans
                    ans_texts = answers.get("text", []) if isinstance(answers, dict) else []

                    qa_items.append({
                        "id": f"tydi_full_{len(qa_items) + 1:05d}",
                        "question": question,
                        "expected_document_key": filename,
                        "document_title": title,
                        "language": lang,
                        "gold_answers": ans_texts,
                        "mode": "hybrid",
                    })

                    if limit is not None and len(files) >= limit:
                        break

                offset += batch_size
                print(f"  Processed {offset} rows | {len(files)} docs | {len(qa_items)} QA items ...")
                if limit is not None and len(files) >= limit:
                    break
        except Exception as exc:
            print(f"  Warning during fetch at offset {offset}: {exc}")
            break

    dataset_file = dest_dir / "tydiqa_full_dataset.json"
    dataset_file.write_text(json.dumps(qa_items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Saved {len(files)} TyDi QA documents to {doc_dir}")
    print(f"✓ Saved {len(qa_items)} ground-truth QA evaluation pairs to {dataset_file}")

    return files, dataset_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and format Google Research TyDi QA dataset")
    parser.add_argument("--dest", default="/tmp/tydiqa_full_corpus", help="Destination folder")
    parser.add_argument("--split", default="validation", help="Dataset split (validation or train)")
    parser.add_argument("--limit", type=int, default=None, help="Document limit (None for full split)")
    args = parser.parse_args()
    fetch_tydiqa_corpus(Path(args.dest), limit=args.limit, split=args.split)
