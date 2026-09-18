"""Node 4: PDF Downloader & Section-Aware Text Parser.

Downloads the target arXiv PDF with local disk caching, parses document structure
into distinct sections (Abstract, Intro, Methods, Results, Limitations, References),
tracks page numbers for grounding citations, and handles corrupted or scanned PDFs gracefully.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import os
import re
import time
from pathlib import Path
import urllib.request
import httpx

from arxiv_digest.config import AgentConfig
from arxiv_digest.models import PaperMetadata, PaperSection, ParsedPaper
from arxiv_digest.state import AgentState

logger = logging.getLogger(__name__)

# Known scientific paper section heading patterns
SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:[0-9IVXLCDM]+\.?\s+)?(?:Abstract|Introduction|Background|Related Work|Methodology|Method|Architecture|System Design|Approach|Experiments|Experimental Setup|Results|Evaluation|Discussion|Limitations|Broader Impacts|Conclusion|Conclusions|References|Bibliography))\b",
    re.IGNORECASE,
)


def download_pdf(
    paper: PaperMetadata,
    cache_dir: Path,
    timeout: float = 45.0,
) -> Path | None:
    """Download arXiv PDF with local disk caching and polite User-Agent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized_id = paper.arxiv_id.replace("/", "_")
    target_path = cache_dir / f"{sanitized_id}.pdf"

    if target_path.exists() and target_path.stat().st_size > 1024:
        logger.info(f"Using cached PDF: {target_path}")
        return target_path

    urls_to_try = [
        paper.pdf_url,
        f"https://arxiv.org/pdf/{paper.arxiv_id}",
        f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf",
    ]

    for url in urls_to_try:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "curl/8.16.0",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()
                if len(content) > 1000:
                    with open(target_path, "wb") as f:
                        f.write(content)
                    return target_path
        except Exception as e:
            logger.debug(f"Failed to download PDF from {url}: {e}")

    logger.warning(f"Could not download PDF for {paper.arxiv_id} from any candidate URL.")
    return None


def extract_with_pymupdf(pdf_path: Path) -> tuple[list[PaperSection], list[str], str]:
    """Extract structured sections and references using PyMuPDF."""
    import pymupdf  # type: ignore

    doc = pymupdf.open(str(pdf_path))
    sections: list[PaperSection] = []
    references: list[str] = []
    full_text_parts: list[str] = []

    current_heading = "Introduction"
    current_content: list[str] = []
    current_start_page = 1

    in_references = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        page_idx = page_num + 1

        for b in blocks:
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            block_text = b[4].strip()
            if not block_text:
                continue

            full_text_parts.append(block_text)
            first_line = block_text.split("\n")[0].strip()

            # Check if this block is a section heading
            if len(first_line) < 60 and SECTION_HEADER_REGEX.match(first_line):
                # Save previous section
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

                current_heading = first_line
                current_start_page = page_idx
                
                if "reference" in first_line.lower() or "bibliography" in first_line.lower():
                    in_references = True
                else:
                    in_references = False

                # Remaining lines in the block go to section content
                remaining_lines = block_text.split("\n")[1:]
                if remaining_lines:
                    current_content.append("\n".join(remaining_lines))
            else:
                if in_references:
                    references.append(block_text)
                else:
                    current_content.append(block_text)

    # Append the final section
    if current_content:
        sections.append(
            PaperSection(
                heading=current_heading,
                content="\n".join(current_content),
                page_start=current_start_page,
                page_end=len(doc),
            )
        )

    doc.close()
    return sections, references, "\n\n".join(full_text_parts)


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
