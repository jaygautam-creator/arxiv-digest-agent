"""Explicit state graph: nodes, edges and a generic runner over a shared AgentState.

The topology is data, not control flow. `build_research_graph()` declares every node and
every edge (with its routing condition); `StateGraph.run()` executes it; and
`StateGraph.to_mermaid()` renders the same definition as the README diagram, so the
documented graph and the executed graph cannot drift apart.

Two entry points share one graph and one state:
  analyze:  query_understanding → … → summarize_briefing → persist_session
  ask:      answer_question → persist_session   (once per QA turn)

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from arxiv_digest.config import AgentConfig
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.nodes.arxiv_client import arxiv_retrieval_node
from arxiv_digest.nodes.chunker import chunk_and_embed_node
from arxiv_digest.nodes.pdf_parser import fetch_and_parse_node
from arxiv_digest.nodes.qa_agent import answer_question_node
from arxiv_digest.nodes.query_parser import parse_query_node
from arxiv_digest.nodes.ranker import paper_ranking_node
from arxiv_digest.nodes.summarizer import summarize_briefing_node
from arxiv_digest.nodes.vector_store import index_chunks_node
from arxiv_digest.state import AgentState

logger = logging.getLogger(__name__)

END = "end"
ERROR = "error"  # terminal state reached when a node records an error

# (node_name, status, message) with status in {"running", "success", "error"}
ProgressCallback = Callable[[str, str, str], None]


@dataclass(frozen=True)
class Context:
    """Dependencies available to every node."""

    config: AgentConfig
    llm: BaseLLM


NodeFn = Callable[[AgentState, Context], AgentState]
Condition = Callable[[AgentState], bool]


@dataclass(frozen=True)
class Node:
    name: str
    run: NodeFn
    description: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    condition: Condition | None = None  # None means "always"
    label: str = ""


class GraphError(RuntimeError):
    """Raised for an invalid graph definition or a routing dead end."""


@dataclass
class StateGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    entry_points: dict[str, str] = field(default_factory=dict)

    def add_node(self, name: str, run: NodeFn, description: str) -> "StateGraph":
        if name in self.nodes or name in (END, ERROR):
            raise GraphError(f"Duplicate or reserved node name: {name}")
        self.nodes[name] = Node(name, run, description)
        return self

    def add_edge(self, source: str, target: str, condition: Condition | None = None, label: str = "") -> "StateGraph":
        self.edges.append(Edge(source, target, condition, label))
        return self

    def add_entry_point(self, name: str, node: str) -> "StateGraph":
        self.entry_points[name] = node
        return self

    def validate(self) -> None:
        """Check that every edge and entry point references a known node and every node has a way out."""
        known = set(self.nodes) | {END, ERROR}
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in known:
                raise GraphError(f"Edge references unknown node: {edge.source} → {edge.target}")
        for entry, node in self.entry_points.items():
            if node not in self.nodes:
                raise GraphError(f"Entry point '{entry}' references unknown node: {node}")
        for name in self.nodes:
            outgoing = [e for e in self.edges if e.source == name]
            if not outgoing:
                raise GraphError(f"Node has no outgoing edge: {name}")
            if all(e.condition is not None for e in outgoing):
                raise GraphError(f"Node '{name}' needs an unconditional fallback edge")

    def next_node(self, current: str, state: AgentState) -> str:
        """Follow the first outgoing edge whose condition holds (edges are checked in declaration order)."""
        for edge in self.edges:
            if edge.source == current and (edge.condition is None or edge.condition(state)):
                return edge.target
        raise GraphError(f"No edge out of '{current}' matched")

    def run(
        self,
        state: AgentState,
        context: Context,
        entry: str,
        on_progress: ProgressCallback | None = None,
        max_steps: int = 50,
    ) -> AgentState:
        """Execute from an entry point until END or ERROR. Any node that records an error routes to ERROR."""
        notify = on_progress or (lambda *_: None)
        current = self.entry_points[entry]
        for _ in range(max_steps):
            node = self.nodes[current]
            errors_before, logs_before = len(state.errors), len(state.execution_logs)
            # Record the step before running it, so a node that saves the state (persist_session)
            # writes a path that includes itself.
            state.current_node = node.name
            state.visited_nodes.append(node.name)
            notify(node.name, "running", node.description)
            state = node.run(state, context)

            if len(state.errors) > errors_before:
                notify(node.name, "error", state.errors[-1])
                state.current_node = ERROR
                return state
            new_logs = state.execution_logs[logs_before:]
            notify(node.name, "success", new_logs[-1].message if new_logs else "done")

            current = self.next_node(current, state)
            if current == END:
                state.current_node = END
                return state
        raise GraphError(f"Exceeded {max_steps} steps; the graph may contain a cycle")

    def reachable_from(self, node: str) -> set[str]:
        """Nodes reachable from `node` (inclusive), following all edges."""
        seen, frontier = set(), [node]
        while frontier:
            current = frontier.pop()
            if current in seen or current not in self.nodes:
                continue
            seen.add(current)
            frontier.extend(e.target for e in self.edges if e.source == current)
        return seen

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid flowchart.

        Nodes reachable from only one entry point are grouped in a subgraph for that entry;
        nodes shared by several entries stay outside. Error routing is drawn once per group.
        """
        reach = {entry: self.reachable_from(node) for entry, node in self.entry_points.items()}
        shared = {n for n in self.nodes if sum(n in r for r in reach.values()) > 1}

        def box(name: str) -> str:
            # A Markdown-string label: node name, then its description on a second line.
            return f'{name}["`{name}\n{self.nodes[name].description}`"]'

        # SVG text labels are measured with the font they are drawn in, so GitHub's renderer
        # doesn't clip them the way it can clip HTML labels.
        lines = ['%%{init: {"flowchart": {"htmlLabels": false}}}%%', "flowchart TD"]
        for entry, first in self.entry_points.items():
            lines.append(f"    {entry}_in([{entry}]) --> {first}")
        for entry, members in reach.items():
            lines.append(f"    subgraph {entry}_flow [{entry}]")
            lines.extend(f"        {box(n)}" for n in self.nodes if n in members and n not in shared)
            lines.append("    end")
        lines.extend(f"    {box(n)}" for n in self.nodes if n in shared)
        for edge in self.edges:
            arrow = f"-- {edge.label} -->" if edge.label else "-->"
            # "end" is a Mermaid keyword, so terminal states get their own ids in the diagram.
            target = "finished([end])" if edge.target == END else edge.target
            lines.append(f"    {edge.source} {arrow} {target}")
        lines.append("    failed([error: stop, errors kept in state])")
        for entry in reach:
            lines.append(f"    {entry}_flow -. any node error .-> failed")
        lines.extend(f"    {n} -. error .-> failed" for n in self.nodes if n in shared)
        return "\n".join(lines)


def is_topic_search(state: AgentState) -> bool:
    return state.intent == "TOPIC_SEARCH"


def build_research_graph() -> StateGraph:
    """Declare the agent's graph. This is the single source of truth for its topology."""
    graph = (
        StateGraph()
        .add_node("query_understanding", lambda s, c: parse_query_node(s), "classify topic vs arXiv ID")
        .add_node("arxiv_retrieval", lambda s, c: arxiv_retrieval_node(s, c.config), "query the arXiv Atom API")
        .add_node("paper_ranking", lambda s, c: paper_ranking_node(s, c.llm), "choose one candidate")
        .add_node("fetch_and_parse", lambda s, c: fetch_and_parse_node(s, c.config), "download PDF, extract sections")
        .add_node("chunk_and_embed", lambda s, c: chunk_and_embed_node(s, c.config), "section-bounded chunks")
        .add_node("vector_indexing", lambda s, c: index_chunks_node(s, c.config), "TF-IDF / hybrid index")
        .add_node("summarize_briefing", lambda s, c: summarize_briefing_node(s, c.llm), "structured briefing")
        .add_node("answer_question", lambda s, c: answer_question_node(s, c.llm, c.config), "grounded RAG answer")
        .add_node("persist_session", _persist_session, "save state as JSON")
        .add_entry_point("analyze", "query_understanding")
        .add_entry_point("ask", "answer_question")
        .add_edge("query_understanding", "arxiv_retrieval")
        .add_edge("arxiv_retrieval", "paper_ranking", is_topic_search, "topic search")
        .add_edge("arxiv_retrieval", "fetch_and_parse", label="direct ID")
        .add_edge("paper_ranking", "fetch_and_parse")
        .add_edge("fetch_and_parse", "chunk_and_embed")
        .add_edge("chunk_and_embed", "vector_indexing")
        .add_edge("vector_indexing", "summarize_briefing")
        .add_edge("summarize_briefing", "persist_session")
        .add_edge("answer_question", "persist_session")
        .add_edge("persist_session", END)
    )
    graph.validate()
    return graph


def _persist_session(state: AgentState, context: Context) -> AgentState:
    path = state.save_session(context.config.sessions_dir)
    state.log_step("persist_session", "success", f"Saved session to {path}")
    return state


class ResearchStateGraph:
    """The agent's graph bound to its configuration and LLM."""

    def __init__(self, config: AgentConfig, llm: BaseLLM):
        self.context = Context(config=config, llm=llm)
        self.graph = build_research_graph()

    def execute(self, query: str, on_progress: ProgressCallback | None = None) -> AgentState:
        """Analyze a topic or arXiv ID: retrieval through briefing."""
        return self.graph.run(AgentState(raw_query=query), self.context, "analyze", on_progress)

    def ask(self, state: AgentState, question: str, on_progress: ProgressCallback | None = None) -> AgentState:
        """Run one QA turn on an analyzed state."""
        state.pending_question = question
        return self.graph.run(state, self.context, "ask", on_progress)
