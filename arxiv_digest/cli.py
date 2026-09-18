"""Command-Line Interface for the arXiv Digest & QA Agent.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arxiv_digest.agent import ArxivDigestAgent
from arxiv_digest.config import AgentConfig, LLMProviderType
from arxiv_digest.llm import ProviderConfigError, describe_provider
from arxiv_digest.state import AgentState

console = Console()


def print_banner() -> None:
    """Render application header banner."""
    banner_text = Text()
    banner_text.append("Autonomous arXiv Paper Digest & QA Agent\n", style="bold cyan")
    banner_text.append("Stateful Research Graph  |  Authored by Jay Gautam for 8byte", style="dim italic")
    console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))


def print_briefing(state: AgentState) -> None:
    """Render the structured executive briefing."""
    if not state.briefing:
        console.print("[bold red]No executive briefing was generated.[/bold red]")
        return

    b = state.briefing
    md_content = b.to_markdown()
    console.print(Panel(Markdown(md_content), title="[bold green]Executive Briefing[/bold green]", border_style="green"))


def run_qa_interactive_loop(agent: ArxivDigestAgent, state: AgentState) -> None:
    """Launch interactive Question-Answering REPL."""
    console.print("\n[bold cyan]Entering Interactive QA Mode[/bold cyan]")
    console.print("[dim]Ask any question regarding the paper. Type 'exit' or 'quit' to finish.[/dim]\n")

    if state.briefing and state.briefing.suggested_followup_questions:
        console.print("[yellow]Suggested Questions to try:[/yellow]")
        for q in state.briefing.suggested_followup_questions:
            console.print(f"  [dim]•[/dim] {q}")
        console.print()

    while True:
        try:
            question = console.input("[bold cyan]Ask Paper > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting QA mode.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[cyan]Retrieving grounded chunks & synthesizing answer...[/cyan]"):
            response = agent.ask(state, question)

        console.print()
        if response.is_grounded:
            console.print(Panel(response.answer, title="[bold green]Answer[/bold green]", border_style="green"))
            
            if response.citations:
                cite_table = Table(title="Retrieved Provenance & Citations", show_header=True, header_style="bold blue")
                cite_table.add_column("Section", style="cyan", width=25)
                cite_table.add_column("Page", justify="center", width=8)
                cite_table.add_column("Excerpt Snippet", style="dim", overflow="fold")

                for c in response.citations:
                    cite_table.add_row(c.section, str(c.page), c.excerpt)
                console.print(cite_table)
        else:
            console.print(Panel(
                f"[yellow]{response.answer}[/yellow]",
                title="[bold yellow]Ungrounded / Out-of-Scope Query[/bold yellow]",
                border_style="yellow",
            ))
        console.print()


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Autonomous arXiv Paper Digest & QA Agent (Jay Gautam for 8byte)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research topic (e.g. 'KV-cache compression') or arXiv ID/URL (e.g. '1706.03762')",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "ollama", "mock"],
        default=None,
        help="LLM provider to use (default: auto-detected from environment)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force deterministic offline mock provider (no API keys required)",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Path to an existing session JSON file to resume QA directly",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        help="Export executive briefing as structured JSON to specified path",
    )
    parser.add_argument(
        "--export-md",
        type=str,
        help="Export executive briefing as Markdown to specified path",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not enter interactive QA loop after generating briefing",
    )

    args = parser.parse_args()

    print_banner()

    config = AgentConfig.from_env()
    if args.mock:
        config.provider = LLMProviderType.MOCK
    elif args.provider:
        config.provider = LLMProviderType(args.provider)

    try:
        agent = ArxivDigestAgent(config=config)
    except ProviderConfigError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(2)

    console.print(f"[bold]LLM Provider:[/bold] [cyan]{describe_provider(config)}[/cyan]")
    if config.provider == LLMProviderType.MOCK:
        console.print(
            "[bold yellow]Mock mode: no LLM is called. Briefing and answers are placeholders; "
            "set GEMINI_API_KEY or GROQ_API_KEY in .env for real output.[/bold yellow]"
        )

    # Resume session or execute graph
    if args.session:
        session_file = Path(args.session)
        if not session_file.exists():
            console.print(f"[bold red]Session file not found:[/bold red] {session_file}")
            sys.exit(1)
        console.print(f"[cyan]Resuming existing session from:[/cyan] {session_file}")
        state = agent.load_session(session_file)
    else:
        if not args.query:
            console.print("[bold red]Error:[/bold red] Please provide a research topic or arXiv ID.")
            parser.print_help()
            sys.exit(1)

        console.print(f"[bold]Target Query:[/bold] [yellow]{args.query}[/yellow]\n")

        with console.status("[bold cyan]Executing state graph pipeline...[/bold cyan]") as status:
            def on_progress(node_name: str, node_status: str, msg: str):
                status.update(f"[cyan][{node_name}][/cyan] {msg}")

            state = agent.analyze(args.query, on_progress=on_progress)

    for warning in state.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    model_used = getattr(agent.llm, "last_model_used", None)
    if model_used and config.provider == LLMProviderType.GEMINI and model_used != config.gemini_model:
        console.print(f"[yellow]Note:[/yellow] {config.gemini_model} was unavailable; answered by {model_used}.")

    # Check for fatal errors
    if state.errors:
        console.print("\n[bold red]Pipeline encountered errors:[/bold red]")
        for err in state.errors:
            console.print(f"  [red]•[/red] {err}")
        if not state.briefing:
            sys.exit(1)

    # Render briefing
    console.print()
    print_briefing(state)

    # Exports
    if args.export_json and state.briefing:
        out_p = Path(args.export_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(state.briefing.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓ Exported JSON briefing to:[/green] {out_p}")

    if args.export_md and state.briefing:
        out_p = Path(args.export_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(state.briefing.to_markdown(), encoding="utf-8")
        console.print(f"[green]✓ Exported Markdown briefing to:[/green] {out_p}")

    # QA loop
    if not args.no_interactive and state.is_complete:
        run_qa_interactive_loop(agent, state)


if __name__ == "__main__":
    main()
