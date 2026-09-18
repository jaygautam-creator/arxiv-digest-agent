"""Retrieval and grounding evaluation over a small hand-labelled question set.

Usage:
    python evals/run_eval.py            # retrieval + similarity gate only (offline once PDFs are cached)
    python evals/run_eval.py --llm      # also score end-to-end answers with the configured LLM
    python evals/run_eval.py --verbose  # list every question, not just misses

Retrieval metrics use exactly what the QA node sees: the top-k chunks that pass the
similarity threshold. A question is a hit when any of its evidence strings appears in
one of those chunks (compared ignoring case, whitespace and table separators, since PDF
text is spaced unpredictably, e.g. "0 . 1").

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from arxiv_digest.config import AgentConfig
from arxiv_digest.llm import get_llm_provider
from arxiv_digest.nodes.arxiv_client import fetch_from_arxiv
from arxiv_digest.nodes.chunker import chunk_and_embed_node
from arxiv_digest.nodes.pdf_parser import fetch_and_parse_node
from arxiv_digest.nodes.qa_agent import answer_question
from arxiv_digest.nodes.vector_store import LocalVectorStore, get_vector_store, index_chunks_node, retrieve
from arxiv_digest.state import AgentState

QUESTIONS = Path(__file__).with_name("questions.json")
ANSWERABLE = ("lexical", "paraphrase")


UNICODE_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2212": "-"})


def normalize(text: str) -> str:
    # Whitespace (including narrow no-break spaces), table separators and dash variants are formatting.
    return re.sub(r"[\s|\u202f\u00a0]+", "", text.translate(UNICODE_DASHES)).lower()


def contains_any(text: str, needles: list[str]) -> bool:
    haystack = normalize(text)
    return any(normalize(n) in haystack for n in needles)


def build_paper_state(arxiv_id: str, config: AgentConfig) -> AgentState:
    """Run the fetch/parse/chunk/index nodes for one paper, exactly as the agent does."""
    papers = fetch_from_arxiv(id_list=arxiv_id, timeout=config.request_timeout)
    if not papers:
        sys.exit(f"Could not fetch arXiv metadata for {arxiv_id}")
    state = AgentState(raw_query=arxiv_id, intent="DIRECT_ID", selected_paper=papers[0])
    for node in (fetch_and_parse_node, chunk_and_embed_node, index_chunks_node):
        state = node(state, config)
        if state.errors:
            sys.exit(f"{arxiv_id}: {state.errors[-1]}")
    return state


def store_for(state: AgentState) -> LocalVectorStore:
    """The index built for this paper's session by the vector_indexing node."""
    store = get_vector_store(state.session_id)
    if store is None:
        sys.exit(f"No index was built for {state.raw_query}")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="also score answers from the configured LLM")
    parser.add_argument("--verbose", action="store_true", help="print every question")
    parser.add_argument("--out", type=Path, help="write per-question results as JSON")
    parser.add_argument("--pause", type=float, default=8.0, help="seconds between LLM calls (free-tier rate limits)")
    args = parser.parse_args()

    spec = json.loads(QUESTIONS.read_text())
    config = AgentConfig.from_env()
    llm = get_llm_provider(config) if args.llm else None

    states = {pid: build_paper_state(pid, config) for pid in spec["papers"]}
    mode = store_for(next(iter(states.values()))).mode
    print(f"retrieval mode: {mode}\n")

    # Guard against a stale question set: every evidence string must exist in its paper's chunks.
    for q in spec["questions"]:
        if q["kind"] in ANSWERABLE:
            corpus = " ".join(c.text for c in states[q["paper"]].chunks)
            if not contains_any(corpus, q["evidence"]):
                sys.exit(f"Evidence not found in parsed paper {q['paper']}: {q['question']}")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    records = []
    for q in spec["questions"]:
        state = states[q["paper"]]
        retrieved = retrieve(store_for(state), q["question"], config)
        kind = q["kind"]
        s = stats[kind]
        s["n"] += 1

        if kind in ANSWERABLE:
            ok = any(contains_any(chunk.text, q["evidence"]) for chunk, _ in retrieved)
            s["hit"] += ok
            s["gated"] += not retrieved
            label = "hit " if ok else ("GATED" if not retrieved else "MISS")
        else:
            ok = not retrieved
            s["gated"] += ok
            label = "gated" if ok else "passed gate"

        record = {
            **q,
            "retrieved": [[c.section_heading, c.page_label, round(sc, 3)] for c, sc in retrieved],
            "retrieval_ok": ok,
        }
        answer_note = ""
        if llm is not None:
            time.sleep(args.pause)
            response = answer_question(q["question"], state, llm, config)
            record.update(answer=response.answer, grounded=response.is_grounded)
            if kind in ANSWERABLE:
                correct = response.is_grounded and contains_any(response.answer, q["answer_any"])
                s["answer_ok"] += correct
                answer_note = " | answer ok" if correct else f" | answer WRONG: {response.answer[:140]!r}"
            else:
                s["refused"] += not response.is_grounded
                answer_note = " | refused" if not response.is_grounded else f" | ANSWERED: {response.answer[:140]!r}"

        records.append(record)
        if args.verbose or not ok or "WRONG" in answer_note or "ANSWERED" in answer_note:
            top = f"{retrieved[0][1]:.3f}" if retrieved else "-"
            print(f"[{label:>11}] {q['paper']} {kind:<12} top={top}  {q['question']}{answer_note}")

    print()
    print(f"{'kind':<13}{'n':>3}  {'retrieval hit@' + str(config.retrieval_top_k):>16}  {'gate refusals':>14}", end="")
    print(f"  {'answers ok / refused':>21}" if llm else "")
    for kind in ("lexical", "paraphrase", "unanswerable", "out_of_scope"):
        s = stats[kind]
        if not s["n"]:
            continue
        hit = f"{s['hit']}/{s['n']}" if kind in ANSWERABLE else "-"
        line = f"{kind:<13}{s['n']:>3}  {hit:>16}  {s['gated']:>11}/{s['n']}"
        if llm:
            done = s["answer_ok"] if kind in ANSWERABLE else s["refused"]
            line += f"  {done:>18}/{s['n']}"
        print(line)

    if args.out:
        args.out.write_text(json.dumps({"mode": mode, "results": records}, indent=2))


if __name__ == "__main__":
    main()
