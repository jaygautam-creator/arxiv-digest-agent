"""Node 3: Candidate Paper Selection & Ranking.

Chooses one paper from the topic-search candidates. The LLM judges relevance from titles,
dates and abstracts (told to prefer newer work when the query asks for recent papers);
if that call fails, a keyword-overlap score with the same recency preference decides.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
import re
import time
from datetime import date

from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.models import PaperMetadata
from arxiv_digest.state import AgentState

RECENCY_CUES = re.compile(
    r"\b(recent|latest|new|newest|current|state[- ]of[- ]the[- ]art|sota|20\d\d)\b", re.IGNORECASE
)


def wants_recent(query: str) -> bool:
    """True when the user's wording asks for recent work."""
    return bool(RECENCY_CUES.search(query))


def recency_score(paper: PaperMetadata, today: date | None = None) -> float:
    """1.0 for a paper published today, decaying linearly to 0 over three years."""
    today = today or date.today()
    try:
        published = date.fromisoformat(paper.published_date[:10])
    except ValueError:
        return 0.0
    age_days = max(0, (today - published).days)
    return max(0.0, 1.0 - age_days / (3 * 365))


def compute_heuristic_score(paper: PaperMetadata, query: str) -> float:
    """Lexical match between query and paper metadata, plus recency when the query asks for it."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return 0.0

    title_words = set(re.findall(r"\w+", paper.title.lower()))
    abstract_words = set(re.findall(r"\w+", paper.abstract.lower()))

    title_overlap = len(query_terms.intersection(title_words)) / len(query_terms)
    abstract_overlap = len(query_terms.intersection(abstract_words)) / len(query_terms)

    # 60% title weight, 40% abstract weight
    score = (title_overlap * 0.6) + (abstract_overlap * 0.4)
    if wants_recent(query):
        score += 0.3 * recency_score(paper)
    return score


def select_paper_with_llm(
    query: str,
    candidates: list[PaperMetadata],
    llm: BaseLLM,
) -> tuple[int, str]:
    """Prompt the LLM to select the most relevant candidate with rationale."""
    options_text = ""
    for idx, paper in enumerate(candidates):
        options_text += (
            f"Candidate [{idx}]:\n"
            f"Title: {paper.title}\n"
            f"Published: {paper.published_date}\n"
            f"Abstract: {paper.abstract[:400]}...\n\n"
        )

    recency_instruction = (
        f"Today is {date.today().isoformat()}. The user asked for recent work, so when candidates are "
        "comparably relevant, prefer the more recently published one."
        if wants_recent(query)
        else ""
    )
    prompt = f"""You are an expert AI research scientist helping rank arXiv papers for a literature review.
The user is researching: "{query}"

Review the following candidate papers:
{options_text}

Select the single candidate that is most directly relevant and impactful for this topic.
{recency_instruction}
Respond in JSON format:
{{
  "selected_index": <int 0 to {len(candidates) - 1}>,
  "rationale": "<1-2 sentence technical reason for selecting this paper over the others>"
}}
"""
    try:
        raw_resp = llm.generate(prompt=prompt, json_mode=True)
        # Parse JSON
        match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            idx = int(data.get("selected_index", 0))
            rationale = str(data.get("rationale", "Selected as best match for query."))
            if 0 <= idx < len(candidates):
                return idx, rationale
    except Exception:
        pass

    # Fallback to heuristic score
    scores = [compute_heuristic_score(p, query) for p in candidates]
    best_idx = int(scores.index(max(scores))) if scores else 0
    return best_idx, "Selected based on highest title and abstract keyword relevance."


def paper_ranking_node(state: AgentState, llm: BaseLLM) -> AgentState:
    """Graph Node: Rank candidate papers and select the primary paper for digestion."""
    start_time = time.time()

    if state.selected_paper is not None:
        state.log_step(
            "paper_ranking", "skipped", "Paper already specified directly.", (time.time() - start_time) * 1000
        )
        return state

    if not state.candidate_papers:
        msg = "No candidate papers available to rank."
        state.add_error(msg)
        state.log_step("paper_ranking", "error", msg, (time.time() - start_time) * 1000)
        return state

    if len(state.candidate_papers) == 1:
        state.selected_paper = state.candidate_papers[0]
        state.selection_rationale = "Single candidate retrieved matching the query."
        msg = f"Automatically selected sole candidate: '{state.selected_paper.title}'"
        state.log_step("paper_ranking", "success", msg, (time.time() - start_time) * 1000)
        return state

    # Rank across candidates
    chosen_idx, rationale = select_paper_with_llm(
        query=state.raw_query,
        candidates=state.candidate_papers,
        llm=llm,
    )

    state.selected_paper = state.candidate_papers[chosen_idx]
    state.selection_rationale = rationale
    position = f"[{chosen_idx + 1}/{len(state.candidate_papers)}]"
    msg = f"Selected paper {position}: '{state.selected_paper.title}' - Rationale: {rationale}"
    state.log_step("paper_ranking", "success", msg, (time.time() - start_time) * 1000)

    return state
