"""Node 2: arXiv API Client.

Interfaces with the official arXiv Atom XML feed to search and retrieve paper metadata.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import urllib.request
import httpx

from arxiv_digest.config import AgentConfig
from arxiv_digest.models import PaperMetadata
from arxiv_digest.state import AgentState

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _clean_text(text: str | None) -> str:
    """Normalize multi-line whitespace and strip text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_atom_entry(entry: ET.Element) -> PaperMetadata:
    """Extract structured PaperMetadata from an Atom XML entry."""
    # Extract ID
    id_el = entry.find("atom:id", ATOM_NS)
    raw_id = id_el.text if id_el is not None and id_el.text else ""
    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

    # Title & Abstract
    title_el = entry.find("atom:title", ATOM_NS)
    title = _clean_text(title_el.text) if title_el is not None else "Untitled"

    summary_el = entry.find("atom:summary", ATOM_NS)
    abstract = _clean_text(summary_el.text) if summary_el is not None else ""

    # Authors
    authors = []
    for author_el in entry.findall("atom:author", ATOM_NS):
        name_el = author_el.find("atom:name", ATOM_NS)
        if name_el is not None and name_el.text:
            authors.append(_clean_text(name_el.text))

    # Dates
    pub_el = entry.find("atom:published", ATOM_NS)
    published = pub_el.text[:10] if pub_el is not None and pub_el.text else "Unknown"

    upd_el = entry.find("atom:updated", ATOM_NS)
    updated = upd_el.text[:10] if upd_el is not None and upd_el.text else None

    # Links
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    for link_el in entry.findall("atom:link", ATOM_NS):
        rel = link_el.attrib.get("rel")
        title_attr = link_el.attrib.get("title")
        href = link_el.attrib.get("href")
        if title_attr == "pdf" and href:
            pdf_url = href.replace("http://", "https://")
        elif rel == "alternate" and href:
            abs_url = href.replace("http://", "https://")

    # Categories
    categories = []
    for cat_el in entry.findall("atom:category", ATOM_NS):
        term = cat_el.attrib.get("term")
        if term:
            categories.append(term)

    prim_el = entry.find("arxiv:primary_category", ATOM_NS)
    primary_category = prim_el.attrib.get("term") if prim_el is not None else (categories[0] if categories else None)

    comment_el = entry.find("arxiv:comment", ATOM_NS)
    comment = _clean_text(comment_el.text) if comment_el is not None else None

    doi_el = entry.find("arxiv:doi", ATOM_NS)
    doi = _clean_text(doi_el.text) if doi_el is not None else None

    # Strip version suffix from canonical ID for clarity
    clean_id = re.sub(r"v\d+$", "", arxiv_id)

    return PaperMetadata(
        arxiv_id=clean_id,
        title=title,
        authors=authors,
        abstract=abstract,
        pdf_url=pdf_url,
        abs_url=abs_url,
        published_date=published,
        updated_date=updated,
        categories=categories,
        primary_category=primary_category,
        comment=comment,
        doi=doi,
    )


def fetch_from_arxiv(
    query: str | None = None,
    id_list: str | None = None,
    max_results: int = 5,
    sort_by: str = "relevance",
    timeout: float = 30.0,
) -> list[PaperMetadata]:
    """Execute request to the official arXiv API Atom feed."""
    params: dict[str, str | int] = {}

    if id_list:
        params["id_list"] = id_list
    elif query:
        params["start"] = 0
        params["max_results"] = max_results
        
        # Stopwords that pollute arXiv Atom boolean queries
        stopwords = {"recent", "work", "on", "for", "the", "a", "an", "in", "of", "and", "to", "with", "paper", "papers", "study", "new", "using"}
        tokens = [t for t in re.findall(r"[a-zA-Z0-9_\-]+", query) if len(t) > 2 and t.lower() not in stopwords]
        
        if not tokens:
            # Fallback to any token
            tokens = [t for t in re.findall(r"[a-zA-Z0-9_\-]+", query) if len(t) > 1]

        if len(tokens) > 1:
            params["search_query"] = " AND ".join(f"all:{t}" for t in tokens[:5])
        elif tokens:
            params["search_query"] = f"all:{tokens[0]}"
        else:
            params["search_query"] = f"all:{query.strip()}"

        params["sortBy"] = sort_by
        params["sortOrder"] = "descending"

    url = f"{ARXIV_API_BASE}?{urlencode(params)}"
    
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "curl/8.16.0",
            "Accept": "*/*",
        },
    )
    
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        xml_content = resp.read()

    root = ET.fromstring(xml_content)
    papers: list[PaperMetadata] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        # Some empty results return an entry with title "Error"
        title_el = entry.find("atom:title", ATOM_NS)
        if title_el is not None and title_el.text and "Error" in title_el.text:
            continue
        try:
            papers.append(parse_atom_entry(entry))
        except Exception as e:
            logger.debug(f"Failed to parse Atom entry: {e}")

    return papers


def arxiv_retrieval_node(state: AgentState, config: AgentConfig) -> AgentState:
    """Graph Node: Query the arXiv API and update candidate papers."""
    start_time = time.time()

    if state.intent == "DIRECT_ID" and state.parsed_arxiv_id:
        try:
            papers = fetch_from_arxiv(id_list=state.parsed_arxiv_id, timeout=config.request_timeout)
            if not papers:
                # Fallback: search via query
                papers = fetch_from_arxiv(query=state.parsed_arxiv_id, max_results=1, timeout=config.request_timeout)

            if papers:
                state.candidate_papers = papers
                state.selected_paper = papers[0]
                state.selection_rationale = f"Direct lookup for arXiv ID {state.parsed_arxiv_id}"
                msg = f"Retrieved exact match for paper: '{papers[0].title}' ({papers[0].arxiv_id})"
                state.log_step("arxiv_retrieval", "success", msg, (time.time() - start_time) * 1000)
            else:
                msg = f"No paper found on arXiv matching ID: {state.parsed_arxiv_id}"
                state.add_error(msg)
                state.log_step("arxiv_retrieval", "error", msg, (time.time() - start_time) * 1000)
        except Exception as e:
            msg = f"arXiv API error during direct ID lookup: {e}"
            state.add_error(msg)
            state.log_step("arxiv_retrieval", "error", msg, (time.time() - start_time) * 1000)

    elif state.intent == "TOPIC_SEARCH":
        try:
            papers = fetch_from_arxiv(
                query=state.raw_query,
                max_results=config.arxiv_max_results,
                timeout=config.request_timeout,
            )
            # Failure case handling: What happens if arXiv returns 0 candidate papers for a vague topic?
            if not papers:
                state.add_warning("Strict keyword search yielded 0 results. Relaxing query to broader terms...")
                # Broader relaxed search using single primary keywords
                words = [w for w in state.raw_query.split() if len(w) > 3]
                if words:
                    relaxed_query = words[0]
                    papers = fetch_from_arxiv(query=relaxed_query, max_results=config.arxiv_max_results, timeout=config.request_timeout)

            if papers:
                state.candidate_papers = papers
                msg = f"Found {len(papers)} candidate paper(s) for topic: '{state.raw_query}'"
                state.log_step("arxiv_retrieval", "success", msg, (time.time() - start_time) * 1000)
            else:
                msg = f"Zero candidate papers found on arXiv for query: '{state.raw_query}'"
                state.add_error(msg)
                state.log_step("arxiv_retrieval", "error", msg, (time.time() - start_time) * 1000)
        except Exception as e:
            msg = f"arXiv API error during topic search: {e}"
            state.add_error(msg)
            state.log_step("arxiv_retrieval", "error", msg, (time.time() - start_time) * 1000)

    return state
