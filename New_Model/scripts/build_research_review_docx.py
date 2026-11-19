"""Build the single consolidated Korean research review with deliberate pages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".document_deps"))

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

WORK = ROOT / "docs/research_review_2026-11-09"
OUTPUT = ROOT / "docs/FedX_MAG_실험_진행방식과_한계_통합검토_2026-11-09.docx"


def font_style(style, size, bold=False):
    style.font.name = "Malgun Gothic"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for name in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{name}"), "Malgun Gothic")
    for name in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme", "csTheme"):
        fonts.attrib.pop(qn(f"w:{name}"), None)
    style.font.complex_script = False


def rich(paragraph, text):
    for index, part in enumerate(re.split(r"\*\*(.*?)\*\*", text)):
        run = paragraph.add_run(part)
        run.bold = bool(index % 2)


def field(paragraph, name):
    element = OxmlElement("w:fldSimple")
    element.set(qn("w:instr"), name)
    paragraph._p.append(element)


def add_table(document, lines):
    rows = [[part.strip() for part in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", value) for value in row)]
    count = len(rows[0])
    if rows[0][0] == "데이터와 후보":
        widths = [57, 19, 19, 17, 54]
    elif rows[0][0] == "구성" and count == 5:
        widths = [66, 30, 30, 20, 20]
    elif rows[0][0] == "근거":
        widths = [14, 97, 55]
    elif count == 3:
        widths = [36, 54, 76]
    elif count == 4:
        widths = [43, 41, 41, 41]
    else:
        widths = [166 / count] * count
    table = document.add_table(rows=0, cols=count)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Mm(width)
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{name}")
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            edge.set(qn(f"w:{key}"), value)
        borders.append(edge)
    table._tbl.tblPr.append(borders)
    for index, values in enumerate(rows):
        row = table.add_row()
        prevent_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(prevent_split)
        if index == 0:
            repeat = OxmlElement("w:tblHeader")
            row._tr.get_or_add_trPr().append(repeat)
        for column, (cell, value) in enumerate(zip(row.cells, values, strict=True)):
            cell.width = Mm(widths[column])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            margin = OxmlElement("w:tcMar")
            for edge, amount in (("top", "75"), ("bottom", "75"), ("left", "90"), ("right", "90")):
                node = OxmlElement(f"w:{edge}")
                node.set(qn("w:w"), amount)
                node.set(qn("w:type"), "dxa")
                margin.append(node)
            properties.append(margin)
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "DFE8EF" if index == 0 else ("F6F8FA" if index % 2 else "FFFFFF"))
            properties.append(shade)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.16
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if index == 0 or (column > 0 and len(value) < 18):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(value)
            run.font.size = Pt(8.8 if rows[0][0] == "근거" else 9.2)
            run.bold = index == 0
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = 0.3
    spacer.add_run().font.size = Pt(3)


def run():
    text = (WORK / "manuscript.md").read_text(encoding="utf-8")
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.top_margin, section.bottom_margin = Mm(19), Mm(18)
    section.left_margin, section.right_margin = Mm(22), Mm(22)
    section.header_distance, section.footer_distance = Mm(8), Mm(8)
    normal = document.styles["Normal"]
    font_style(normal, 10.3)
    normal.paragraph_format.line_spacing = 1.22
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Title", 22), ("Heading 1", 17), ("Heading 2", 11.7)):
        style = document.styles[name]
        font_style(style, size, True)
        style.paragraph_format.space_before = Pt(12 if name == "Heading 2" else 0)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.line_spacing = 1.2
        borders = style.element.get_or_add_pPr().find(qn("w:pBdr"))
        if borders is not None:
            borders.getparent().remove(borders)
    header = section.header.paragraphs[0]
    header.text = "FedX MAG 구조 개선 실험 검토"
    for item in header.runs:
        item.font.size = Pt(8)
        item.font.color.rgb = RGBColor(0, 0, 0)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("2026년 11월 9일 기준   ·   ")
    field(footer, "PAGE")
    footer.add_run(" / ")
    footer.add_run("10")
    for item in footer.runs:
        item.font.size = Pt(8)
    lines = text.splitlines()
    index = 0
    new_page = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!--page-->":
            new_page = True
        elif line.startswith("|"):
            chosen = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                chosen.append(lines[index])
                index += 1
            add_table(document, chosen)
            continue
        elif line.startswith("# "):
            document.add_paragraph(line[2:], "Title")
        elif line.startswith("## "):
            paragraph = document.add_paragraph(line[3:], "Heading 1")
            paragraph.paragraph_format.page_break_before = new_page
            new_page = False
        elif line.startswith("### "):
            document.add_paragraph(line[4:], "Heading 2")
        else:
            paragraph = document.add_paragraph()
            rich(paragraph, line)
        index += 1
    document.core_properties.title = "FedX MAG 실험 진행 방식과 한계"
    document.core_properties.subject = "실험 방법과 검증 상태 및 연구 운영 한계의 통합 검토"
    document.core_properties.author = "Codex"
    document.core_properties.keywords = "FedX MAG, 이상 탐지, 실험 검토, 한계, 앙상블"
    document.save(OUTPUT)
    print(json.dumps({"docx": str(OUTPUT), "planned_pages": 10}, ensure_ascii=False))


if __name__ == "__main__":
    run()
