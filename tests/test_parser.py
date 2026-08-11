import tempfile
import unittest
from pathlib import Path

from supabase_easy_rag.ingestion.parser import (
    checksum_for_text,
    extract_metadata_block,
    extract_sections,
    first_heading,
    parse_markdown_document,
)


class TestParser(unittest.TestCase):
    def test_first_heading_extraction(self):
        md = "# Main Title\n\nSome text content."
        self.assertEqual(first_heading(md, "fallback.md"), "Main Title")

    def test_extract_metadata_block(self):
        md = """# Document

## Metadata
- **Classification**: Internal
- **Category**: Tech
- **Author**: Alex
---

## Body
Content goes here.
"""
        meta = extract_metadata_block(md)
        self.assertEqual(meta.get("classification"), "Internal")
        self.assertEqual(meta.get("category"), "Tech")
        self.assertEqual(meta.get("author"), "Alex")

    def test_extract_sections(self):
        md = """# Doc Title

## Section 1
Content 1

### Subsection 1.1
Content 1.1

## Section 2
Content 2
"""
        sections = extract_sections(md)
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0].heading, "Section 1")
        self.assertEqual(sections[0].level, 2)
        self.assertEqual(sections[1].heading, "Subsection 1.1")
        self.assertEqual(sections[1].level, 3)
        self.assertEqual(sections[1].parent_section_id, sections[0].id)
        self.assertEqual(sections[2].heading, "Section 2")

    def test_checksum_calculation(self):
        text = "Hello World"
        chk1 = checksum_for_text(text)
        chk2 = checksum_for_text(text)
        self.assertEqual(chk1, chk2)
        self.assertEqual(len(chk1), 64)

    def test_parse_markdown_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            doc_dir = tmp_path / "docs" / "tech"
            doc_dir.mkdir(parents=True)
            doc_file = doc_dir / "sample.md"
            doc_file.write_text("# Tech Document\n\n## Metadata\n- **Category**: Software\n---\n\n## Overview\nSample text", encoding="utf-8")

            parsed = parse_markdown_document(doc_file, tmp_path)
            self.assertEqual(parsed.title, "Tech Document")
            self.assertEqual(parsed.top_level_category, "docs")
            self.assertEqual(len(parsed.sections), 1)
            self.assertEqual(parsed.metadata.get("category"), "Software")


if __name__ == "__main__":
    unittest.main()
