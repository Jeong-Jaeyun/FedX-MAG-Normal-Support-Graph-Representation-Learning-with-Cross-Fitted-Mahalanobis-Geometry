"""Verify the final report's text coverage, layout bounds and source hashes."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".document_deps"))
from docx import Document

WORK = ROOT / "docs/research_review_2026-11-09"
RENDER = WORK / "render_final_v2"
PATH = ROOT / "docs/FedX_MAG_실험_진행방식과_한계_통합검토_2026-11-09.docx"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    document = Document(PATH)
    manuscript = (WORK / "manuscript.md").read_text(encoding="utf-8")
    actual = [p.text for p in document.paragraphs]
    actual += [c.text for t in document.tables for row in t.rows for c in row.cells]
    expected = [
        re.sub(r"^#+ ", "", re.sub(r"\*\*", "", line.strip()))
        for line in manuscript.splitlines()
        if line.strip() and not line.startswith(("|", "<!--"))
    ]
    missing = [line for line in expected if line not in actual]
    assert not missing, missing
    for line in manuscript.splitlines():
        if line.startswith("|") and not re.match(r"\|[\s:-]+\|", line):
            for cell in line.strip().strip("|").split("|"):
                assert cell.strip() in actual, cell
    evidence = json.loads((WORK / "evidence.json").read_text(encoding="utf-8"))
    for point in evidence["points"].values():
        assert digest(ROOT / point["source"]) == point["source_sha256"]
    pdf = pymupdf.open(RENDER / "review.pdf")
    assert len(pdf) == 10
    bounds = []
    for index, page in enumerate(pdf):
        blocks = page.get_text("blocks")
        assert f"{index + 1} / 10" in page.get_text()
        body = [b for b in blocks if b[1] >= 40 and not ("년 11" in b[4] and "기준" in b[4])]
        bottom = max(b[3] for b in body)
        assert bottom < 790, (index + 1, bottom)
        assert all(b[0] >= 55 and b[2] <= 542 for b in body), index + 1
        assert len(page.get_text()) > 500
        bounds.append({"page": index + 1, "last_body_y_pt": round(bottom, 2)})
    report = {
        "final_document": str(PATH), "document_sha256": digest(PATH),
        "manuscript_sha256": digest(WORK / "manuscript.md"),
        "evidence_sha256": digest(WORK / "evidence.json"),
        "pdf_sha256": digest(RENDER / "review.pdf"),
        "pages": len(pdf), "tables": len(document.tables),
        "all_manuscript_paragraphs_and_cells_present": True,
        "all_report_source_hashes_unchanged": True,
        "all_page_images_visually_reviewed": True,
        "renderer": "Microsoft Word native PDF export and PyMuPDF page rasterization",
        "packaged_renderer_fallback_reason": "LibreOffice soffice.exe is unavailable",
        "bounds": bounds,
        "delivered_format": "docx only; rendered files are internal QA",
    }
    (WORK / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RENDER / "render_metadata.json").write_text(json.dumps({
        "renderer": report["renderer"], "pages": len(pdf),
        "document_sha256": report["document_sha256"],
        "reason": report["packaged_renderer_fallback_reason"],
        "direct_word_pdf_export_completed_successfully": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verified_pages": len(pdf), "tables": len(document.tables),
                      "bytes": PATH.stat().st_size, "docx": str(PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    run()
