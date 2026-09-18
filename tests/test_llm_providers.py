"""Tests for provider selection, retry/fallback behaviour and honest summarization fallback.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
from unittest.mock import patch

import httpx
import pytest

from arxiv_digest.config import AgentConfig, LLMProviderType, load_dotenv
from arxiv_digest.llm import ProviderConfigError, get_llm_provider
from arxiv_digest.llm.base import LLMError, post_with_retry
from arxiv_digest.llm.gemini_client import GeminiLLM
from arxiv_digest.models import PaperMetadata
from arxiv_digest.nodes.summarizer import degraded_briefing, parse_briefing_json
from arxiv_digest.state import AgentState

ENV_VARS = ["LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MOCK_LLM", "USE_OLLAMA"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)  # isolate from any real .env in the repo
    return tmp_path


def _state() -> AgentState:
    state = AgentState(raw_query="2604.24971")
    state.selected_paper = PaperMetadata(
        arxiv_id="2604.24971",
        title="Real Title",
        authors=["Real Author"],
        abstract="The real abstract.",
        pdf_url="https://arxiv.org/pdf/2604.24971",
        abs_url="https://arxiv.org/abs/2604.24971",
        published_date="2026-04-27",
    )
    return state


def test_dotenv_is_loaded_and_placeholders_ignored(clean_env):
    (clean_env / ".env").write_text("GEMINI_API_KEY=your_gemini_api_key_here\nGROQ_API_KEY=gsk_real\n")
    config = AgentConfig.from_env()
    assert config.gemini_api_key is None
    assert config.provider == LLMProviderType.GROQ


def test_dotenv_does_not_override_real_environment(clean_env, monkeypatch):
    (clean_env / ".env").write_text("GROQ_API_KEY=from_file\n")
    monkeypatch.setenv("GROQ_API_KEY", "from_shell")
    load_dotenv()
    assert AgentConfig.from_env().groq_api_key == "from_shell"


def test_no_keys_means_mock(clean_env):
    assert AgentConfig.from_env().provider == LLMProviderType.MOCK


def test_explicit_provider_without_key_is_an_error(clean_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    config = AgentConfig.from_env()
    with pytest.raises(ProviderConfigError):
        get_llm_provider(config)


def _response(status: int, body: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body or {}, request=httpx.Request("POST", "https://x"))


def test_retry_recovers_from_overload():
    responses = iter([_response(503), _response(200, {"ok": True})])
    with patch("httpx.Client.post", side_effect=lambda *a, **k: next(responses)), patch("time.sleep"):
        assert post_with_retry("https://x", {}) == {"ok": True}


def test_client_errors_are_not_retried():
    with patch("httpx.Client.post", return_value=_response(400)) as post, patch("time.sleep"), pytest.raises(LLMError):
        post_with_retry("https://x", {})
    assert post.call_count == 1


def test_gemini_falls_back_to_next_model():
    ok = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    def fake_post(url, payload, **kwargs):
        if "primary" in url:
            raise LLMError("HTTP 503")
        return ok

    llm = GeminiLLM(api_key="k", model_name="primary", fallback_models=["backup"])
    with patch("arxiv_digest.llm.gemini_client.post_with_retry", side_effect=fake_post):
        assert llm.generate("hi") == "hello"
    assert llm.last_model_used == "backup"


def test_briefing_metadata_always_comes_from_arxiv():
    llm_output = {
        "title": "Hallucinated Title",
        "authors": ["Invented Person"],
        "arxiv_id": "0000.00000",
        "publish_date": "1999-01-01",
        "link": "https://example.com",
        "summary_plain_english": "s",
        "problem_statement": "p",
        "method_approach": ["m"],
        "key_results_claims": ["r"],
        "limitations": ["l"],
        "suggested_followup_questions": ["q"],
    }
    briefing = parse_briefing_json(json.dumps(llm_output), _state())
    assert briefing.title == "Real Title"
    assert briefing.authors == ["Real Author"]
    assert briefing.publish_date == "2026-04-27"


def test_unparseable_briefing_raises():
    with pytest.raises(ValueError):
        parse_briefing_json("not json at all", _state())


def test_degraded_briefing_invents_nothing():
    briefing = degraded_briefing(_state(), "the LLM summarization step failed")
    assert briefing.summary_plain_english == "The real abstract."
    for field in (briefing.key_results_claims, briefing.method_approach, briefing.limitations):
        assert all(item.startswith("Not available") for item in field)


def test_gemini_skips_failed_model_for_rest_of_session():
    ok = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    calls = []

    def fake_post(url, payload, **kwargs):
        calls.append(url)
        if "primary" in url:
            raise LLMError("HTTP 503")
        return ok

    llm = GeminiLLM(api_key="k", model_name="primary", fallback_models=["backup"])
    with patch("arxiv_digest.llm.gemini_client.post_with_retry", side_effect=fake_post):
        llm.generate("first")
        llm.generate("second")
    assert sum("primary" in url for url in calls) == 1


def test_recency_only_counts_when_query_asks_for_it():
    from datetime import date

    from arxiv_digest.nodes.ranker import compute_heuristic_score, recency_score, wants_recent

    def paper(published: str) -> PaperMetadata:
        return PaperMetadata(
            arxiv_id="x",
            title="KV cache compression",
            abstract="KV cache compression for LLMs",
            pdf_url="u",
            abs_url="u",
            published_date=published,
        )

    assert wants_recent("recent work on KV-cache compression")
    assert not wants_recent("KV-cache compression")
    assert recency_score(paper("2026-09-01"), today=date(2026, 9, 18)) > recency_score(
        paper("2024-10-04"), today=date(2026, 9, 18)
    )

    new, old = paper(date.today().isoformat()), paper("2015-01-01")
    assert compute_heuristic_score(new, "recent KV cache compression") > compute_heuristic_score(
        old, "recent KV cache compression"
    )
    assert compute_heuristic_score(new, "KV cache compression") == compute_heuristic_score(old, "KV cache compression")


def test_auto_prefers_groq_when_both_keys_present(clean_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert AgentConfig.from_env().provider == LLMProviderType.GROQ


def test_error_messages_are_short():
    body = {"error": {"code": 429, "message": "You exceeded your current quota.\nDetails follow..."}}
    with (
        patch("httpx.Client.post", return_value=_response(429, body)),
        patch("time.sleep"),
        pytest.raises(LLMError) as exc,
    ):
        post_with_retry("https://x", {}, max_attempts=1)
    assert "You exceeded your current quota." in str(exc.value)
    assert "Details follow" not in str(exc.value)


def test_summarizer_reasks_once_on_malformed_json():
    from arxiv_digest.llm.base import BaseLLM
    from arxiv_digest.models import PaperSection, ParsedPaper
    from arxiv_digest.nodes.summarizer import summarize_briefing_node

    valid = {
        "summary_plain_english": "s",
        "problem_statement": "p",
        "method_approach": ["m"],
        "key_results_claims": ["r"],
        "limitations": ["l"],
        "suggested_followup_questions": ["q"],
    }

    class FlakyLLM(BaseLLM):
        calls = 0

        def generate(self, prompt, system_prompt=None, json_mode=False):
            self.calls += 1
            return '{"method_approach": [["nested"]]}' if self.calls == 1 else json.dumps(valid)

    state = _state()
    state.parsed_paper = ParsedPaper(
        metadata=state.selected_paper,
        sections=[PaperSection(heading="1 Introduction", content="text")],
        raw_text="text",
    )
    llm = FlakyLLM()
    state = summarize_briefing_node(state, llm)
    assert llm.calls == 2
    assert state.briefing.problem_statement == "p"
    assert not state.warnings


def test_prose_only_drops_flattened_tables_and_labels():
    from arxiv_digest.nodes.summarizer import prose_only

    text = "\n\n".join(
        [
            "GRKV raises the average score from 33.96 to 34.58 with SnapKV.",
            "SnapKV 3.24 59.00 23.20 4.00 1.40 17.00 15.10 78.40 54.20 27.44 0/13",
            "niah_mk1",
        ]
    )
    assert prose_only(text) == "GRKV raises the average score from 33.96 to 34.58 with SnapKV."
