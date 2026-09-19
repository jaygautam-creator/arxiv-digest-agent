"""Node 4: PDF Downloader & Section-Aware Text Parser.

Downloads the target arXiv PDF with local disk caching, parses document structure
into distinct sections (Abstract, Intro, Methods, Results, Limitations, References),
tracks page numbers for grounding citations, and handles corrupted or scanned PDFs gracefully.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

from arxiv_digest.config import USER_AGENT, AgentConfig
from arxiv_digest.models import PaperMetadata, PaperSection, ParsedPaper
from arxiv_digest.state import AgentState

logger = logging.getLogger(__name__)

# Section names recognised by the pypdf fallback, which has no font information to go on.
KNOWN_SECTION_NAMES = (
    "Abstract", "Introduction", "Background", "Related Work", "Methodology", "Method", "Architecture",
    "System Design", "Approach", "Experiments", "Experimental Setup", "Results", "Evaluation", "Discussion",
    "Limitations", "Broader Impacts", "Conclusion", "Conclusions", "References", "Bibliography",
)  # fmt: skip
SECTION_HEADER_REGEX = re.compile(
    r"^(?:[0-9IVXLCDM]+\.?\s+)?(?:" + "|".join(KNOWN_SECTION_NAMES) + r")\b",
    re.IGNORECASE,
)


def is_pdf(data: bytes) -> bool:
    """PDF files start with "%PDF-"; arXiv error or rate-limit pages are HTML."""
    return data[:5] == b"%PDF-"


def download_pdf(
    paper: PaperMetadata,
    cache_dir: Path,
    timeout: float = 45.0,
) -> Path | None:
    """Download the paper's PDF, with a local cache that only ever holds real PDFs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized_id = paper.arxiv_id.replace("/", "_")
    target_path = cache_dir / f"{sanitized_id}.pdf"

    if target_path.exists():
        with open(target_path, "rb") as f:
            if is_pdf(f.read(5)):
                logger.info(f"Using cached PDF: {target_path}")
                return target_path
        logger.warning(f"Cached file {target_path} is not a PDF; downloading again.")
        target_path.unlink()

    urls_to_try = [
        paper.pdf_url,
        f"https://arxiv.org/pdf/{paper.arxiv_id}",
        f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf",
    ]

    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()
        except Exception as e:
            logger.debug(f"Failed to download PDF from {url}: {e}")
            continue
        if is_pdf(content):
            target_path.write_bytes(content)
            return target_path
        logger.debug(f"Response from {url} is not a PDF ({len(content)} bytes); skipping.")

    logger.warning(f"Could not download PDF for {paper.arxiv_id} from any candidate URL.")
    return None


# A numbered heading label: "3", "3.2", "3.2.1", or an appendix label "A", "B.1".
SECTION_NUMBER_REGEX = re.compile(r"^(?:\d{1,2}|[A-H])(?:\.\d{1,2}){0,2}\.?$")
NUMBERED_HEADING_REGEX = re.compile(r"^((?:\d{1,2}|[A-H])(?:\.\d{1,2}){0,2})\.?\s+([A-Z].{1,80})$")
# Words that signal a heading title wraps onto the next line.
CONTINUATION_ENDINGS = ("and", "of", "for", "the", "with", "a", "an", "to", "in", "on", "via", "through", "-", ",", ":")
REFERENCE_START_REGEX = re.compile(r"^\[\d{1,3}\]")
UNNUMBERED_HEADINGS = {
    "abstract",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "limitations",
    "conclusion",
    "conclusions",
    "discussion",
    "broader impact",
}


def _is_bold(span: dict) -> bool:
    font = span.get("font", "")
    return bool(span.get("flags", 0) & 16) or any(tag in font for tag in ("Bold", "Medi", "Semibold", ".B"))


class Line(NamedTuple):
    page: int
    block: int
    text: str
    bold: bool
    size: float
    is_table_row: bool = False  # a reconstructed table row: cells joined by " | "


ROW_Y_TOLERANCE = 3.0  # points; cells of one table row sit on (almost) the same baseline
HEADER_SEARCH_HEIGHT = 150.0  # points above a table block searched for column headers
NUMERIC_CELL = re.compile(r"^[\d.,%×x±+\-–/()GMKB]+$")


class Geometry(NamedTuple):
    block: int
    y0: float
    y1: float
    x0: float
    x1: float
    text: str


def _table_rows(geometry: list[Geometry]) -> list[list[tuple[float, str]]] | None:
    """Rebuild rows of (x_center, cell) from one block's lines, or None if it is not a table.

    In LaTeX PDFs each table cell is usually its own line, so a table block has several
    lines side by side on the same baseline; prose has one line per baseline.
    """
    rows: list[list[Geometry]] = []
    for g in sorted(geometry, key=lambda g: (g.y0, g.x0)):
        if rows and abs(g.y0 - rows[-1][0].y0) <= ROW_Y_TOLERANCE:
            rows[-1].append(g)
        else:
            rows.append([g])
    if max(len(r) for r in rows) < 3:
        return None

    table = []
    for row in rows:
        cells: list[tuple[float, str]] = []
        for g in sorted(row, key=lambda g: g.x0):
            tokens = g.text.split()
            if len(tokens) > 1 and all(NUMERIC_CELL.match(t) for t in tokens):
                # PDF generators sometimes merge adjacent numeric cells into one line;
                # split it and spread the cells evenly across the line's width.
                step = (g.x1 - g.x0) / len(tokens)
                cells.extend((g.x0 + step * (k + 0.5), t) for k, t in enumerate(tokens))
            else:
                cells.append(((g.x0 + g.x1) / 2, g.text))
        table.append(cells)
    return table


def _column_header(x_center: float, table_top: float, candidates: list[Geometry]) -> str | None:
    """The lowest short text above the table whose horizontal extent covers this column."""
    covering = [
        g
        for g in candidates
        if g.x0 - 3 <= x_center <= g.x1 + 3 and table_top - HEADER_SEARCH_HEIGHT <= g.y0 < table_top
    ]
    return max(covering, key=lambda g: g.y0).text if covering else None


def _render_table(
    rows: list[list[tuple[float, str]]], table_top: float, page_geometry: list[Geometry], block: int
) -> list[str]:
    """Render rows as "label | header: value | ...", attaching column headers found above the table."""
    # A column header sits over one column; text spanning three or more columns is a group
    # label (e.g. "Llama-3.1-8B-Instruct, 10% Cache Budget"), not a header.
    column_centers = [x for x, _ in max(rows, key=len)[1:]]
    candidates = [
        g
        for g in page_geometry
        if g.block != block
        and len(g.text) <= 40
        and any(ch.isalpha() for ch in g.text)
        and not NUMERIC_CELL.match(g.text.replace(" ", ""))
        and sum(g.x0 - 3 <= x <= g.x1 + 3 for x in column_centers) < 3
    ]
    rendered = []
    for cells in rows:
        is_data_row = any(NUMERIC_CELL.match(text) for _, text in cells[1:])
        parts = [cells[0][1]]
        for x, text in cells[1:]:
            header = _column_header(x, table_top, candidates) if is_data_row else None
            parts.append(f"{header}: {text}" if header else text)
        rendered.append(" | ".join(parts))
    return rendered


def _read_lines(doc) -> tuple[list[Line], float]:
    """Flatten the document into lines plus the body font size; table blocks become one line per row."""
    lines: list[Line] = []
    size_weight: dict[float, int] = {}
    for page_idx, page in enumerate(doc, start=1):
        page_blocks: list[tuple[int, list[Line], list[Geometry]]] = []
        for block_idx, block in enumerate(page.get_text("dict")["blocks"]):
            block_lines: list[Line] = []
            geometry: list[Geometry] = []
            for line in block.get("lines", []):
                spans = [sp for sp in line["spans"] if sp["text"].strip()]
                if not spans:
                    continue
                text = " ".join(sp["text"].strip() for sp in spans)
                size = round(max(sp["size"] for sp in spans), 1)
                block_lines.append(Line(page_idx, block_idx, text, all(_is_bold(sp) for sp in spans), size))
                x0, y0, x1, y1 = line["bbox"]
                geometry.append(Geometry(block_idx, y0, y1, x0, x1, text))
                size_weight[size] = size_weight.get(size, 0) + len(text)
            page_blocks.append((block_idx, block_lines, geometry))

        page_geometry = [g for _, _, geometry in page_blocks for g in geometry]
        for block_idx, block_lines, geometry in page_blocks:
            rows = _table_rows(geometry) if len(geometry) >= 3 else None
            if rows is None:
                lines.extend(block_lines)
                continue
            table_top = min(g.y0 for g in geometry)
            for text in _render_table(rows, table_top, page_geometry, block_idx):
                lines.append(Line(page_idx, block_idx, text, False, block_lines[0].size, True))
    body_size = max(size_weight, key=lambda size: size_weight[size]) if size_weight else 10.0
    return lines, body_size


def detect_heading(
    lines, i: int, body_size: float, allow_letter_labels: bool = False
) -> tuple[str, str | None, int] | None:
    """Return (label, number, lines_consumed) if lines[i] starts a section heading.

    LaTeX papers typically render headings in bold at or above body size, with the
    number and title either on one line ("3.2 Attention") or on two ("3.2" / "Attention").
    Requiring bold + body size filters out bold figure labels and table cells. Letter
    labels ("A", "B.1") are only accepted once the appendix can have started.
    """
    text, bold, size = lines[i].text, lines[i].bold, lines[i].size
    if not bold or size < body_size - 0.6 or len(text) > 90:
        return None

    def is_number(label: str) -> bool:
        return bool(SECTION_NUMBER_REGEX.match(label)) and (allow_letter_labels or label[0].isdigit())

    def with_continuation(title: str, j: int) -> tuple[str, int]:
        """Append wrapped title lines (bold, same size) while the title looks unfinished."""
        extra = 0
        while (
            title.lower().endswith(CONTINUATION_ENDINGS)
            and j + extra < len(lines)
            and lines[j + extra].bold
            and abs(lines[j + extra].size - size) < 0.6
            and len(lines[j + extra].text) <= 80
        ):
            nxt = lines[j + extra].text
            title = title[:-1] + nxt if title.endswith("-") else f"{title} {nxt}"
            extra += 1
        return title, extra

    if is_number(text.rstrip(".")) and i + 1 < len(lines):
        nxt, nxt_bold, nxt_size = lines[i + 1].text, lines[i + 1].bold, lines[i + 1].size
        if nxt_bold and nxt_size >= body_size - 0.6 and nxt[:1].isupper() and len(nxt) <= 80:
            title, extra = with_continuation(nxt, i + 2)
            return title, text.rstrip("."), 2 + extra

    m = NUMBERED_HEADING_REGEX.match(text)
    if m and is_number(m.group(1)) and not text.rstrip().endswith((".", ",")):
        title, extra = with_continuation(m.group(2).strip(), i + 1)
        return title, m.group(1), 1 + extra

    if text.lower().rstrip(":") in UNNUMBERED_HEADINGS:
        return text.rstrip(":"), None, 1
    return None


def extract_with_pymupdf(pdf_path: Path) -> tuple[list[PaperSection], list[str], str]:
    """Extract sections (with subsection paths), references, and full text using PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    lines, body_size = _read_lines(doc)
    page_count = len(doc)
    doc.close()

    sections: list[PaperSection] = []
    references: list[str] = []
    ref_block: list[str] = []
    ref_block_id: tuple[int, int] | None = None

    heading, start_page = "Front Matter", 1
    content: list[str] = []  # one paragraph per PDF block
    content_pages: list[int] = []  # page of each paragraph, for exact citations
    content_block: tuple[int, int] | None = None
    top_level = ""  # e.g. "3 Methods", used to prefix subsection headings
    in_references = False
    seen_references = False

    def flush(end_page: int) -> None:
        if content:
            sections.append(
                PaperSection(
                    heading=heading,
                    content="\n\n".join(content),
                    page_start=start_page,
                    page_end=end_page,
                    paragraph_pages=list(content_pages),
                )
            )

    def add_line(text: str, block_id: tuple[int, int], is_table_row: bool) -> None:
        nonlocal content_block
        if content and block_id == content_block:
            prev = content[-1]
            if is_table_row:
                content[-1] = f"{prev}\n{text}"  # keep one table row per line
            elif prev.endswith("-") and text[:1].islower():
                content[-1] = prev[:-1] + text  # re-join words hyphenated across line breaks
            else:
                content[-1] = f"{prev} {text}"
        else:
            content.append(text)
            content_pages.append(block_id[0])
        content_block = block_id

    i = 0
    while i < len(lines):
        page, block, text = lines[i].page, lines[i].block, lines[i].text
        found = detect_heading(lines, i, body_size, allow_letter_labels=seen_references)
        if found:
            label, number, consumed = found
            flush(page)
            content, content_pages, content_block = [], [], None
            if number and "." in number:
                heading = f"{top_level} › {number} {label}" if top_level else f"{number} {label}"
            else:
                top_level = f"{number} {label}" if number else label
                heading = top_level
            start_page = page
            in_references = label.lower() in ("references", "bibliography")
            seen_references = seen_references or in_references
            i += consumed
            continue

        if in_references:
            # A new entry starts at a "[n]" marker; for author-year styles, at a new PDF block.
            numbered = REFERENCE_START_REGEX.match(text) is not None
            new_block = ref_block_id not in (None, (page, block)) and not REFERENCE_START_REGEX.match(
                ref_block[0] if ref_block else ""
            )
            if ref_block and (numbered or new_block):
                references.append(" ".join(ref_block))
                ref_block = []
            ref_block.append(text)
            ref_block_id = (page, block)
        else:
            add_line(text, (page, block), lines[i].is_table_row)
        i += 1

    flush(page_count)
    if ref_block:
        references.append(" ".join(ref_block))

    full_text = "\n".join(line.text for line in lines)
    return sections, references, full_text


def extract_with_pypdf(pdf_path: Path) -> tuple[list[PaperSection], list[str], str]:
    """Fallback text extractor using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    sections: list[PaperSection] = []
    full_text_parts: list[str] = []

    current_heading = "Main Text"
    current_content: list[str] = []
    current_start_page = 1

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        page_idx = idx + 1
        if not text.strip():
            continue

        full_text_parts.append(text)
        lines = text.split("\n")
        for line in lines:
            trimmed = line.strip()
            if SECTION_HEADER_REGEX.match(trimmed) and len(trimmed) < 60:
                if current_content:
                    sections.append(
                        PaperSection(
                            heading=current_heading,
                            content="\n".join(current_content),
                            page_start=current_start_page,
                            page_end=page_idx,
                        )
                    )
                    current_content = []
                current_heading = trimmed
                current_start_page = page_idx
            else:
                current_content.append(trimmed)

    if current_content:
        sections.append(
            PaperSection(
                heading=current_heading,
                content="\n".join(current_content),
                page_start=current_start_page,
                page_end=len(reader.pages),
            )
        )

    return sections, [], "\n\n".join(full_text_parts)


def parse_pdf_document(
    pdf_path: Path | None,
    paper: PaperMetadata,
) -> ParsedPaper:
    """Parse PDF into structured sections with fallback handling for unextractable text."""
    if not pdf_path or not pdf_path.exists():
        # Fallback: create sections from abstract
        return ParsedPaper(
            metadata=paper,
            sections=[
                PaperSection(
                    heading="Abstract",
                    content=paper.abstract,
                    page_start=1,
                    page_end=1,
                )
            ],
            references=[],
            raw_text=paper.abstract,
            parse_warning="PDF file unavailable; fallback to metadata abstract.",
        )

    sections: list[PaperSection] = []
    references: list[str] = []
    raw_text = ""
    parse_warning = None

    try:
        sections, references, raw_text = extract_with_pymupdf(pdf_path)
    except Exception as e:
        logger.warning(f"PyMuPDF failed ({e}), attempting pypdf fallback...")
        try:
            sections, references, raw_text = extract_with_pypdf(pdf_path)
        except Exception as e2:
            parse_warning = f"PDF extraction failed ({e2}); utilizing metadata abstract."
            sections = [PaperSection(heading="Abstract", content=paper.abstract, page_start=1, page_end=1)]
            raw_text = paper.abstract

    # Handle scanned/empty PDF edge case
    if len(raw_text.strip()) < 300:
        parse_warning = "Extracted text was unusually short or scanned; supplemented with arXiv abstract."
        raw_text = f"Abstract:\n{paper.abstract}\n\n" + raw_text
        sections.insert(0, PaperSection(heading="Abstract", content=paper.abstract, page_start=1, page_end=1))

    # Ensure Abstract is always explicitly indexed
    has_abstract = any("abstract" in s.heading.lower() for s in sections)
    if not has_abstract and paper.abstract:
        sections.insert(0, PaperSection(heading="Abstract", content=paper.abstract, page_start=1, page_end=1))

    return ParsedPaper(
        metadata=paper,
        sections=sections,
        references=references[:30],
        raw_text=raw_text,
        parse_warning=parse_warning,
    )


def fetch_and_parse_node(state: AgentState, config: AgentConfig) -> AgentState:
    """Graph Node: Download the chosen paper PDF and parse into structured sections."""
    start_time = time.time()

    if state.selected_paper is None:
        msg = "No paper selected to fetch and parse."
        state.add_error(msg)
        state.log_step("fetch_and_parse", "error", msg, (time.time() - start_time) * 1000)
        return state

    paper = state.selected_paper
    pdf_path = download_pdf(paper=paper, cache_dir=config.cache_dir, timeout=config.request_timeout)

    if pdf_path:
        state.pdf_local_path = str(pdf_path)
    else:
        state.add_warning(f"Could not download PDF directly from {paper.pdf_url}. Falling back to arXiv abstract.")

    parsed = parse_pdf_document(pdf_path, paper)
    state.parsed_paper = parsed

    if parsed.parse_warning:
        state.add_warning(parsed.parse_warning)

    msg = f"Parsed paper into {len(parsed.sections)} sections and {len(parsed.raw_text)} characters."
    state.log_step("fetch_and_parse", "success", msg, (time.time() - start_time) * 1000)

    return state
