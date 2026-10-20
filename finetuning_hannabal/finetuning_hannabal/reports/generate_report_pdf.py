"""Render the Hannibal V2 supervisor report from its Markdown source.

The renderer intentionally supports only the small, explicit Markdown subset
used by this report.  It has no network dependencies and uses ReportLab so the
PDF can be reproduced on the Windows development workstation.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable

from reportlab import rl_config
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


rl_config.invariant = 1

REPORT_DATE = "17 August 2026"
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1E5A7A")
TEAL = colors.HexColor("#0F8B8D")
TEAL_LIGHT = colors.HexColor("#DDF3F2")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#F2F7FA")
LINE = colors.HexColor("#CBD5E1")
WHITE = colors.white
AMBER = colors.HexColor("#F59E0B")
RED = colors.HexColor("#B42318")
GREEN = colors.HexColor("#147D64")


def register_fonts() -> tuple[str, str, str, str]:
    """Register common Windows fonts with portable ReportLab fallbacks."""
    font_dir = Path("C:/Windows/Fonts")
    candidates = {
        "HannibalSans": font_dir / "arial.ttf",
        "HannibalSans-Bold": font_dir / "arialbd.ttf",
        "HannibalSans-Italic": font_dir / "ariali.ttf",
        "HannibalMono": font_dir / "consola.ttf",
    }
    if all(path.is_file() for path in candidates.values()):
        for name, path in candidates.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        addMapping("HannibalSans", 0, 0, "HannibalSans")
        addMapping("HannibalSans", 1, 0, "HannibalSans-Bold")
        addMapping("HannibalSans", 0, 1, "HannibalSans-Italic")
        return "HannibalSans", "HannibalSans-Bold", "HannibalSans-Italic", "HannibalMono"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"


SANS, SANS_BOLD, SANS_ITALIC, MONO = register_fonts()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=SANS,
            fontSize=9.4,
            leading=13.6,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "small": ParagraphStyle(
            "Small", fontName=SANS, fontSize=7.7, leading=10.5, textColor=MUTED, spaceAfter=4
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=SANS_BOLD,
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=11,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=SANS_BOLD,
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=base["Heading3"],
            fontName=SANS_BOLD,
            fontSize=10.5,
            leading=13,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            fontName=SANS_BOLD,
            fontSize=28,
            leading=33,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            fontName=SANS,
            fontSize=14,
            leading=19,
            textColor=TEAL_LIGHT,
            alignment=TA_LEFT,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", fontName=SANS, fontSize=10, leading=15, textColor=INK, alignment=TA_LEFT
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=SANS_ITALIC,
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName=MONO,
            fontSize=6.8,
            leading=9.2,
            textColor=colors.HexColor("#E6EDF3"),
            leftIndent=0,
            rightIndent=0,
        ),
        "table": ParagraphStyle(
            "TableCell", fontName=SANS, fontSize=7.3, leading=9.3, textColor=INK
        ),
        "table_header": ParagraphStyle(
            "TableHeader", fontName=SANS_BOLD, fontSize=7.3, leading=9.3, textColor=WHITE
        ),
        "callout": ParagraphStyle(
            "Callout",
            fontName=SANS,
            fontSize=8.5,
            leading=12,
            textColor=INK,
            leftIndent=4,
            rightIndent=4,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            fontName=SANS_BOLD,
            fontSize=22,
            leading=26,
            textColor=NAVY,
            spaceAfter=15,
        ),
    }


STYLES = build_styles()


def inline(text: str) -> str:
    escaped = html.escape(text.strip())
    # The report uses explicit break markers inside compact callouts.
    escaped = escaped.replace("&lt;br/&gt;", "<br/>").replace("&lt;br /&gt;", "<br/>")
    escaped = re.sub(r"`([^`]+)`", rf'<font name="{MONO}">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    return escaped


def arrow(drawing: Drawing, x1: float, y1: float, x2: float, y2: float, color=TEAL) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.7))
    angle = 5
    if abs(x2 - x1) >= abs(y2 - y1):
        points = [x2, y2, x2 - angle, y2 + 3, x2 - angle, y2 - 3]
    else:
        points = [x2, y2, x2 - 3, y2 + angle, x2 + 3, y2 + angle]
    drawing.add(Polygon(points, fillColor=color, strokeColor=color))


def box(drawing: Drawing, x: float, y: float, w: float, h: float, title: str, subtitle: str = "", fill=PALE) -> None:
    drawing.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=LINE, strokeWidth=1))
    drawing.add(String(x + w / 2, y + h - 15, title, fontName=SANS_BOLD, fontSize=8.6, fillColor=NAVY, textAnchor="middle"))
    if subtitle:
        drawing.add(String(x + w / 2, y + 9, subtitle, fontName=SANS, fontSize=6.5, fillColor=MUTED, textAnchor="middle"))


def architecture_diagram() -> Drawing:
    d = Drawing(475, 260)
    d.add(Rect(0, 0, 475, 260, rx=8, ry=8, fillColor=colors.white, strokeColor=LINE))
    box(d, 18, 188, 88, 45, "User request", "facts and intent", TEAL_LIGHT)
    box(d, 139, 188, 88, 45, "Agent 1", "information only")
    box(d, 260, 188, 105, 45, "Agent 2", "legal reasoning", colors.HexColor("#DCEBFA"))
    box(d, 397, 188, 60, 45, "Agent 3", "formatting")
    arrow(d, 106, 210, 137, 210)
    arrow(d, 227, 210, 258, 210)
    arrow(d, 365, 210, 395, 210)
    core = [
        (20, "Analyze"), (94, "Choose indexes"), (180, "RAG retrieve"),
        (266, "Select clauses"), (352, "Build"), (414, "Validate / risk"),
    ]
    d.add(String(18, 162, "Agent 2 internal sequence", fontName=SANS_BOLD, fontSize=8.5, fillColor=BLUE))
    for i, (x, label) in enumerate(core):
        width = 64 if i not in (1, 5) else 78
        box(d, x, 105, width, 35, label, "", PALE)
        if i < len(core) - 1:
            next_x = core[i + 1][0]
            arrow(d, x + width, 122, next_x - 2, 122)
    stores = [(52, "Blueprints"), (165, "Contracts / Q&A"), (290, "Legal clauses"), (392, "Risk index")]
    for x, label in stores:
        box(d, x, 25, 80, 35, label, "documentary evidence", colors.HexColor("#FFF7E6"))
        arrow(d, x + 40, 60, x + 40, 102, AMBER)
    d.add(String(237, 7, "Legal content comes from retrieved documentary indexes, never from model memory.", fontName=SANS_ITALIC, fontSize=7, fillColor=RED, textAnchor="middle"))
    return d


def pipeline_diagram() -> Drawing:
    d = Drawing(475, 330)
    d.add(Rect(0, 0, 475, 330, rx=8, ry=8, fillColor=colors.white, strokeColor=LINE))
    rows = [
        [(16, "Raw corpus", "PDF / DOC / DOCX"), (125, "Inventory", "hash + duplicates"), (234, "Extraction", "ordered text"), (343, "Canonical JSON", "evidence + audit")],
        [(16, "Qwen packets", "labels only"), (125, "Enriched JSON", "separate artifact"), (234, "Runtime record", "compact template"), (343, "Agent 1 JSON", "facts; no drafting")],
        [(16, "Agent 2 intake", "IDs removed"), (125, "RAG evidence", "template + clauses"), (234, "Target builder", "grounded output"), (343, "Alpaca row", "3 strict fields")],
        [(70, "QLoRA handoff", "audit + tokenize + split"), (197, "Agent 2 adapter", "LoRA weights"), (324, "Ollama package", "merged GGUF Q4_K_M")],
    ]
    ys = [265, 190, 115, 35]
    for row_index, row in enumerate(rows):
        y = ys[row_index]
        for i, (x, title, subtitle) in enumerate(row):
            w = 94 if row_index < 3 else 110
            box(d, x, y, w, 43, title, subtitle, TEAL_LIGHT if row_index in (0, 3) else PALE)
            if i < len(row) - 1:
                arrow(d, x + w, y + 21, row[i + 1][0] - 2, y + 21)
        if row_index < len(rows) - 1:
            end_x = row[-1][0] + (94 if row_index < 3 else 110) / 2
            next_x = rows[row_index + 1][0][0] + (94 if row_index + 1 < 3 else 110) / 2
            d.add(Line(end_x, y, end_x, y - 17, strokeColor=TEAL, strokeWidth=1.5))
            d.add(Line(end_x, y - 17, next_x, y - 17, strokeColor=TEAL, strokeWidth=1.5))
            arrow(d, next_x, y - 17, next_x, ys[row_index + 1] + 45)
    return d


def eligibility_diagram() -> Drawing:
    d = Drawing(475, 225)
    d.add(Rect(0, 0, 475, 225, rx=8, ry=8, fillColor=colors.white, strokeColor=LINE))
    box(d, 18, 160, 100, 42, "Validated artifact", "schema + evidence", TEAL_LIGHT)
    box(d, 188, 160, 100, 42, "Eligibility gate", "mode and reviews", colors.HexColor("#FFF7E6"))
    arrow(d, 118, 181, 186, 181)
    box(d, 355, 160, 102, 42, "Export decision", "manifested status", PALE)
    arrow(d, 288, 181, 353, 181)
    box(d, 40, 55, 155, 60, "Demo branch", "demo_unreviewed\ntraining_eligible: false", colors.HexColor("#FFF1F0"))
    box(d, 280, 55, 155, 60, "Production branch", "approved reviews + volume\ntraining_eligible: true", colors.HexColor("#E8F5EE"))
    d.add(Line(238, 160, 238, 135, strokeColor=TEAL, strokeWidth=1.5))
    d.add(Line(238, 135, 117, 135, strokeColor=TEAL, strokeWidth=1.5))
    d.add(Line(238, 135, 357, 135, strokeColor=TEAL, strokeWidth=1.5))
    arrow(d, 117, 135, 117, 117)
    arrow(d, 357, 135, 357, 117)
    d.add(String(117, 35, "Useful for reproducibility; cannot establish legal readiness.", fontName=SANS_ITALIC, fontSize=7, fillColor=RED, textAnchor="middle"))
    d.add(String(357, 35, "Requires exact review hashes and at least 200 eligible examples.", fontName=SANS_ITALIC, fontSize=7, fillColor=GREEN, textAnchor="middle"))
    return d


DIAGRAMS = {
    "three-agent-architecture": (architecture_diagram, "Figure 1. Hannibal V2 three-agent architecture and the law-from-RAG boundary."),
    "data-pipeline": (pipeline_diagram, "Figure 2. Repository data path from source contracts to the Agent 2 adapter handoff."),
    "eligibility-gates": (eligibility_diagram, "Figure 3. Separation between demonstrator outputs and production eligibility."),
}


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="body",
        )
        self.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=self._page))
        self._heading_counter = 0

    def beforeDocument(self) -> None:
        # ``multiBuild`` performs more than one layout pass so the table of
        # contents can resolve page numbers. Bookmark keys must be identical
        # on every pass or ReportLab treats the TOC as perpetually changing.
        self._heading_counter = 0
        super().beforeDocument()

    def _page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setTitle("Hannibal V2 — Contract Data Pipeline and Fine-Tuning Readiness")
        canvas.setAuthor("E-Tafakna — Hannibal V2")
        canvas.setSubject("Repository technical report")
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.6)
            canvas.line(18 * mm, A4[1] - 16 * mm, A4[0] - 18 * mm, A4[1] - 16 * mm)
            canvas.setFont(SANS_BOLD, 7.4)
            canvas.setFillColor(NAVY)
            canvas.drawString(18 * mm, A4[1] - 12.5 * mm, "HANNIBAL V2")
            canvas.setFont(SANS, 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 12.5 * mm, "Contract Data Pipeline & Fine-Tuning Readiness")
            canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
            canvas.drawString(18 * mm, 9.5 * mm, "E-Tafakna • Technical Report • 17 August 2026")
            canvas.drawRightString(A4[0] - 18 * mm, 9.5 * mm, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Heading1", "Heading2", "Heading3"}:
            level = {"Heading1": 0, "Heading2": 1, "Heading3": 2}[flowable.style.name]
            text = flowable.getPlainText()
            key = f"heading-{self._heading_counter}"
            self._heading_counter += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def cover_flowables() -> list[Flowable]:
    title = Paragraph("HANNIBAL V2", STYLES["cover_title"])
    subtitle = Paragraph("Contract Data Pipeline and<br/>Fine-Tuning Readiness Report", STYLES["cover_subtitle"])
    banner = Table([[title], [subtitle]], colWidths=[174 * mm], rowHeights=[48 * mm, 42 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14 * mm),
        ("TOPPADDING", (0, 0), (-1, 0), 18 * mm),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 1.5, TEAL),
    ]))
    meta = Table([
        [Paragraph("Organization", STYLES["small"]), Paragraph("E-Tafakna", STYLES["cover_meta"])],
        [Paragraph("System", STYLES["small"]), Paragraph("Hannibal V2 — AI Legal Contract Reasoning Engine", STYLES["cover_meta"])],
        [Paragraph("Repository scope", STYLES["small"]), Paragraph("Source-grounded contract ingestion, runtime preparation, Agent 1/Agent 2 data handoff, and adapter preparation", STYLES["cover_meta"])],
        [Paragraph("Report date", STYLES["small"]), Paragraph(REPORT_DATE, STYLES["cover_meta"])],
    ], colWidths=[36 * mm, 132 * mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    doctrine = Table([[Paragraph("Core doctrine: legal content comes exclusively from documentary indexes retrieved through RAG. Fine-tuning improves reasoning and drafting; it is not a source of law.", STYLES["callout"]) ]], colWidths=[168 * mm])
    doctrine.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return [Spacer(1, 15 * mm), banner, Spacer(1, 25 * mm), meta, Spacer(1, 22 * mm), doctrine, PageBreak()]


def toc_flowables() -> list[Flowable]:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName=SANS_BOLD, fontSize=9.4, leading=14, textColor=NAVY, leftIndent=0, firstLineIndent=0, spaceBefore=3),
        ParagraphStyle("TOC2", fontName=SANS, fontSize=8.3, leading=12, textColor=INK, leftIndent=12, firstLineIndent=0),
        ParagraphStyle("TOC3", fontName=SANS, fontSize=7.5, leading=10.5, textColor=MUTED, leftIndent=25, firstLineIndent=0),
    ]
    return [Paragraph("Table of Contents", STYLES["toc_title"]), toc, PageBreak()]


def make_table(rows: list[list[str]]) -> Flowable:
    if not rows:
        return Spacer(1, 1)
    columns = len(rows[0])
    usable = 174 * mm
    widths = [usable / columns] * columns
    converted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = STYLES["table_header"] if row_index == 0 else STYLES["table"]
        converted.append([Paragraph(inline(cell), style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def code_block(text: str) -> Flowable:
    pre = Preformatted(text.rstrip(), STYLES["code"], maxLineLength=110)
    table = Table([[pre]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#172B3A")),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return KeepTogether([table, Spacer(1, 5)])


def callout(lines: list[str]) -> Flowable:
    text = "<br/>".join(inline(line.lstrip("> ")) for line in lines)
    first = " ".join(lines)
    accent = AMBER if "AMBIGUITY" in first or "WARNING" in first else TEAL
    fill = colors.HexColor("#FFF8E8") if accent == AMBER else TEAL_LIGHT
    table = Table([[Paragraph(text, STYLES["callout"]) ]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([table, Spacer(1, 5)])


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        index += 1
    return rows, index


def list_flowable(items: list[str], ordered: bool) -> Flowable:
    list_items = [ListItem(Paragraph(inline(item), STYLES["body"]), leftIndent=12) for item in items]
    return ListFlowable(
        list_items,
        bulletType="1" if ordered else "bullet",
        start="1",
        leftIndent=18,
        bulletFontName=SANS,
        bulletFontSize=8,
        bulletColor=TEAL,
        spaceAfter=5,
    )


def parse_markdown(source: str) -> list[Flowable]:
    lines = source.splitlines()
    story: list[Flowable] = []
    index = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(part.strip() for part in paragraph)), STYLES["body"]))
            paragraph.clear()

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped == "[[COVER]]":
            flush_paragraph()
            story.extend(cover_flowables())
            index += 1
            continue
        if stripped == "[[TOC]]":
            flush_paragraph()
            story.extend(toc_flowables())
            index += 1
            continue
        if stripped == "<!-- PAGEBREAK -->":
            flush_paragraph()
            # Keep major sections visually separated when the current page is
            # already substantially occupied, without forcing every section
            # onto a new page and inflating the report with unused whitespace.
            story.append(CondPageBreak(72 * mm))
            index += 1
            continue
        diagram_match = re.fullmatch(r"\[\[DIAGRAM:([a-z0-9-]+)\]\]", stripped)
        if diagram_match:
            flush_paragraph()
            factory, caption = DIAGRAMS[diagram_match.group(1)]
            story.extend([factory(), Paragraph(caption, STYLES["caption"])])
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            story.append(code_block("\n".join(code_lines)))
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            match = re.match(r"^(#{1,3})\s+(.*)$", stripped)
            if match:
                level = len(match.group(1))
                story.append(Paragraph(inline(match.group(2)), STYLES[f"h{level}"]))
                index += 1
                continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip())
                index += 1
            story.append(callout(quote_lines))
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            flush_paragraph()
            rows, index = parse_table(lines, index)
            story.extend([make_table(rows), Spacer(1, 7)])
            continue
        unordered = re.match(r"^-\s+(.*)$", stripped)
        ordered = re.match(r"^\d+\.\s+(.*)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            is_ordered = ordered is not None
            items: list[str] = []
            pattern = r"^\d+\.\s+(.*)$" if is_ordered else r"^-\s+(.*)$"
            while index < len(lines):
                item_match = re.match(pattern, lines[index].strip())
                if not item_match:
                    break
                items.append(item_match.group(1))
                index += 1
            story.append(list_flowable(items, is_ordered))
            continue
        if stripped == "---":
            flush_paragraph()
            story.append(Spacer(1, 3))
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def build_pdf(source_path: Path, output_path: Path) -> None:
    source_text = source_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=18 * mm,
        title="Hannibal V2 — Contract Data Pipeline and Fine-Tuning Readiness Report",
        author="E-Tafakna — Hannibal V2",
    )
    doc.multiBuild(parse_markdown(source_text))


def build_parser() -> argparse.ArgumentParser:
    report_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=report_dir / "Hannibal_V2_Technical_Report.md")
    parser.add_argument("--output", type=Path, default=report_dir / "Hannibal_V2_Technical_Report.pdf")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Report source does not exist: {args.source}")
    build_pdf(args.source.resolve(), args.output.resolve())
    print(f"Generated {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
