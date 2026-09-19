"""Node 1: Query Understanding & Intent Parsing.

Parses user query into either an exact arXiv ID/URL or a topic search term.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import re
import time

from arxiv_digest.state import AgentState

# Modern arXiv IDs (2401.12345, 1706.03762v2) and legacy IDs (cs/0101001, math.GT/0309136),
# bare or inside an arxiv.org/abs|pdf URL (with or without scheme and "www.").
ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv:)?(?:(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/)?"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def parse_query_node(state: AgentState) -> AgentState:
    """Classify user intent: specific paper lookup vs natural-language topic search."""
    start_time = time.time()
    query = state.raw_query.strip()

    if not query:
        state.add_error("Empty query provided.")
        state.log_step("query_understanding", "error", "Query string was empty.", (time.time() - start_time) * 1000)
        return state

    match = ARXIV_ID_PATTERN.search(query)
    # If the query is predominantly an arXiv ID or URL
    lowered = query.lower()
    is_link = "arxiv.org/" in lowered or lowered.startswith("arxiv:")
    if match and (len(match.group(0)) >= len(query) * 0.7 or is_link):
        arxiv_id = match.group(1)
        # Strip version suffix if present for canonical lookup
        canonical_id = re.sub(r"v\d+$", "", arxiv_id)
        state.intent = "DIRECT_ID"
        state.parsed_arxiv_id = canonical_id
        msg = f"Detected direct arXiv paper ID: '{canonical_id}'"
        state.log_step("query_understanding", "success", msg, (time.time() - start_time) * 1000)
    else:
        state.intent = "TOPIC_SEARCH"
        state.parsed_arxiv_id = None
        # Clean query for search
        cleaned_search = re.sub(r"[^\w\s\-\+\"]", " ", query).strip()
        msg = f"Detected topic search intent: '{cleaned_search}'"
        state.log_step("query_understanding", "success", msg, (time.time() - start_time) * 1000)

    return state
