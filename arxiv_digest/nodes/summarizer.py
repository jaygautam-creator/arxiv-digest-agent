"""Node 6: Executive Briefing Synthesizer.

Generates the comprehensive structured briefing (JSON and Markdown)
strictly adhering to the 7 required rubric dimensions.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
import logging
import re
import time

from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.models import ExecutiveBriefing
from arxiv_digest.state import AgentState, StepStatus

logger = logging.getLogger(__name__)

# Appendix headings are lettered ("B Additional Results", "C.1 ..."); the main body is enough for a briefing.
APPENDIX_HEADING = re.compile(r"^[A-H](?:\.\d+)*\s")


CAPTION = re.compile(r"^(Table|Figure|Fig\.)\s*\d+", re.IGNORECASE)


def is_table_like(paragraph: str, threshold: float = 0.4) -> bool:
    """True for table text: a reconstructed table, or a paragraph where numeric tokens dominate."""
    if " | " in paragraph:
        return True  # reconstructed table rows ("label | header: value | ...")
    tokens = paragraph.split()
    if len(tokens) < 8:
        return False
    numeric = sum(1 for t in tokens if re.fullmatch(r"[\d.,%×x()↑↓+\-–/]+", t))
    return numeric / len(tokens) >= threshold


def is_float_debris(paragraph: str) -> bool:
    """Captions and short labels (e.g. "Llama-3.1-8B, 16K RULER, 10% Cache Budget").

    Tables and figures float in the PDF's reading order, so their captions and group labels
    often land inside unrelated prose and pull its numbers toward the wrong benchmark.
    """
    text = paragraph.strip()
    if CAPTION.match(text):
        return True
    is_sentence = ". " in text or text.endswith((".", "?", "!", ":", "-"))  # "-": a word split mid-sentence
    return len(text.split()) < 12 and not is_sentence


def prose_only(text: str) -> str:
    """Keep running prose: drop tables, captions and labels, and re-join words they split.

    A floating table can interrupt a sentence mid-word ("We evalu-" … table … "ate under"),
    so after removing it, a paragraph ending in a hyphen is joined to a lowercase continuation.
    """
    kept: list[str] = []
    for paragraph in text.split("\n\n"):
        if len(paragraph.split()) < 4 or is_table_like(paragraph) or is_float_debris(paragraph):
            continue
        if kept and kept[-1].endswith("-") and paragraph[:1].islower():
            kept[-1] = kept[-1][:-1] + paragraph
        else:
            kept.append(paragraph)
    return "\n\n".join(kept)


def build_summarization_context(state: AgentState, max_chars: int = 14000) -> str:
    """Assemble the most critical sections of the paper into an executive context window."""
    paper = state.selected_paper
    parsed = state.parsed_paper
    if not paper or not parsed:
        return ""

    context_parts = [
        f"Title: {paper.title}",
        f"Authors: {', '.join(paper.authors)}",
        f"arXiv ID: {paper.arxiv_id}",
        f"Published: {paper.published_date}",
        f"Abstract:\n{paper.abstract}\n",
    ]

    # Prioritize the sections a briefing needs. Headings carry their parent path
    # ("5 Results › 5.2 ..."), so subsections match via their parent's name.
    priority_keywords = [
        "intro",
        "method",
        "approach",
        "architect",
        "design",
        "result",
        "experiment",
        "eval",
        "limit",
        "discuss",
        "conclu",
    ]
    main_body = [s for s in parsed.sections if not APPENDIX_HEADING.match(s.heading)]
    selected_sections = (
        [s for s in main_body if any(kw in s.heading.lower() for kw in priority_keywords)]
        or main_body[:4]
        or parsed.sections[:4]
    )

    # Share the budget fairly so a long introduction cannot crowd out results and limitations;
    # budget a short section leaves unused rolls over to the ones after it.
    remaining_budget = max_chars - sum(len(p) for p in context_parts)
    for idx, sec in enumerate(selected_sections):
        share = remaining_budget // (len(selected_sections) - idx)
        if share < 200:
            break
        sec_text = prose_only(sec.content)[:share].strip()
        context_parts.append(f"--- Section: {sec.heading} (pp. {sec.page_start}-{sec.page_end}) ---\n{sec_text}\n")
        remaining_budget -= len(sec_text)

    return "\n\n".join(context_parts)


UNAVAILABLE = "Not available: {reason}. See the abstract above and use QA mode for details."


def _with_arxiv_metadata(data: dict, state: AgentState) -> dict:
    """Overwrite identifying fields with arXiv's metadata; the LLM is never trusted for these."""
    paper = state.selected_paper
    if paper:
        data.update(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            publish_date=paper.published_date,
            link=paper.abs_url,
        )
    return data


def parse_briefing_json(raw_json: str, state: AgentState) -> ExecutiveBriefing:
    """Parse the LLM response into an ExecutiveBriefing; raises ValueError if it is unusable."""
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        return ExecutiveBriefing.model_validate(_with_arxiv_metadata(data, state))
    except Exception as e:
        raise ValueError(f"LLM returned an unusable briefing: {e}") from e


def degraded_briefing(state: AgentState, reason: str) -> ExecutiveBriefing:
    """Metadata-and-abstract briefing used when the LLM fails.

    Every generated field says plainly that it is unavailable instead of inventing content.
    """
    paper = state.selected_paper
    missing = UNAVAILABLE.format(reason=reason)
    data = _with_arxiv_metadata(
        {
            "summary_plain_english": paper.abstract if paper and paper.abstract else missing,
            "problem_statement": missing,
            "method_approach": [missing],
            "key_results_claims": [missing],
            "limitations": [missing],
            "suggested_followup_questions": [
                "What problem does this paper address?",
                "What are the main results?",
                "What limitations do the authors acknowledge?",
            ],
        },
        state,
    )
    return ExecutiveBriefing.model_validate(data)


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0][:200]


def _generate_briefing(llm: BaseLLM, prompt: str, system_prompt: str, state: AgentState) -> ExecutiveBriefing:
    """Ask for the briefing; if the JSON is malformed, re-ask once with the validation error."""
    raw_output = llm.generate(prompt=prompt, system_prompt=system_prompt, json_mode=True)
    try:
        return parse_briefing_json(raw_output, state)
    except ValueError as e:
        logger.warning("Briefing JSON invalid (%s); asking the model to correct it.", _first_line(e))
        retry_prompt = (
            f"{prompt}\n\nYour previous response did not match the schema: {e}\n"
            "Return only the JSON object, with exactly the keys shown and every list containing plain strings."
        )
        raw_output = llm.generate(prompt=retry_prompt, system_prompt=system_prompt, json_mode=True)
        return parse_briefing_json(raw_output, state)


def summarize_briefing_node(state: AgentState, llm: BaseLLM) -> AgentState:
    """Graph Node: Generate structured executive briefing."""
    start_time = time.time()

    if state.selected_paper is None or state.parsed_paper is None:
        msg = "Missing paper metadata or parsed text to summarize."
        state.add_error(msg)
        state.log_step("summarize_briefing", "error", msg, (time.time() - start_time) * 1000)
        return state

    context = build_summarization_context(state)
    paper = state.selected_paper

    prompt = f"""You are a principal AI researcher preparing an executive briefing for busy engineering leaders.
Synthesize the provided research paper into a high-signal, rigorous briefing.
Report only numbers stated in the text, and attribute each one to the exact method, model and benchmark
the text gives for it. If that attribution is not explicit, describe the result without the number.

PAPER CONTENT:
{context}

Respond ONLY in valid JSON conforming to this schema:
{{
  "title": "{paper.title}",
  "authors": {json.dumps(paper.authors)},
  "arxiv_id": "{paper.arxiv_id}",
  "publish_date": "{paper.published_date}",
  "link": "{paper.abs_url}",
  "summary_plain_english": "<1 paragraph: why this paper matters and its practical significance to an engineer>",
  "problem_statement": "<Precise description of the specific bottleneck, failure mode, or gap addressed>",
  "method_approach": [
    "<Technical bullet 1 on the core mechanism/architecture>",
    "<Technical bullet 2 on implementation/algorithm>",
    "<Technical bullet 3 on optimization or training strategy>"
  ],
  "key_results_claims": [
    "<Empirical result 1 with numbers/benchmarks>",
    "<Empirical result 2 with comparative baselines>"
  ],
  "limitations": [
    "<Explicit technical limitation 1 - DO NOT SKIP THIS>",
    "<Explicit technical limitation 2 regarding assumptions or scaling>"
  ],
  "suggested_followup_questions": [
    "<Critical follow-up question 1>",
    "<Critical follow-up question 2>",
    "<Critical follow-up question 3>"
  ]
}}
"""

    system_prompt = "You are a rigorous AI research scientist. You always return valid JSON and never skip limitations."
    status: StepStatus
    try:
        state.briefing = _generate_briefing(llm, prompt, system_prompt, state)
        status, msg = "success", f"Generated executive briefing for '{state.briefing.title}'."
    except Exception as e:
        # Degrade to an honest abstract-only briefing so metadata and QA remain usable.
        logger.warning("Summarization failed (%s); using abstract-only briefing.", _first_line(e))
        state.briefing = degraded_briefing(state, "the LLM summarization step failed")
        state.add_warning(
            f"Summarization failed ({_first_line(e)}); briefing contains arXiv metadata and abstract only."
        )
        status, msg = "warning", "Summarization failed; produced abstract-only briefing."

    state.is_complete = True
    state.log_step("summarize_briefing", status, msg, (time.time() - start_time) * 1000)
    return state
