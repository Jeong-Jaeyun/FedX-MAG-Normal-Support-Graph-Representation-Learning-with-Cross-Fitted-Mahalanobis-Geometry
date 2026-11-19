"""Rasterize the native Word output and list page boundaries for visual QA."""

import json
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "docs/research_review_2026-11-09/render_final_v2"
document = fitz.open(WORK / "review.pdf")
pages = []
for index, page in enumerate(document):
    image = page.get_pixmap(matrix=fitz.Matrix(1.9, 1.9), alpha=False)
    image.save(WORK / f"page-{index + 1}.png")
    lines = page.get_text().splitlines()
    pages.append({
        "page": index + 1, "lines": len(lines),
        "first": lines[2:5], "last": lines[-5:],
    })
(WORK / "page_text.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"page_count": len(document), "pages": pages}, ensure_ascii=False))
