"""Tests for Query Understanding & Intent Parsing.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import pytest
from arxiv_digest.nodes.query_parser import parse_query_node
from arxiv_digest.state import AgentState


def test_parse_direct_arxiv_id():
    state = AgentState(raw_query="2401.12345")
    updated = parse_query_node(state)
    assert updated.intent == "DIRECT_ID"
    assert updated.parsed_arxiv_id == "2401.12345"
    assert not updated.errors


def test_parse_arxiv_id_with_version():
    state = AgentState(raw_query="1706.03762v7")
    updated = parse_query_node(state)
    assert updated.intent == "DIRECT_ID"
    assert updated.parsed_arxiv_id == "1706.03762"


def test_parse_arxiv_abs_url():
    state = AgentState(raw_query="https://arxiv.org/abs/2401.12345")
    updated = parse_query_node(state)
    assert updated.intent == "DIRECT_ID"
    assert updated.parsed_arxiv_id == "2401.12345"


def test_parse_arxiv_pdf_url():
    state = AgentState(raw_query="https://arxiv.org/pdf/2309.06180.pdf")
    updated = parse_query_node(state)
    assert updated.intent == "DIRECT_ID"
    assert updated.parsed_arxiv_id == "2309.06180"


def test_parse_topic_query():
    state = AgentState(raw_query="recent work on KV-cache compression for LLMs")
    updated = parse_query_node(state)
    assert updated.intent == "TOPIC_SEARCH"
    assert updated.parsed_arxiv_id is None
    assert not updated.errors


def test_parse_empty_query():
    state = AgentState(raw_query="   ")
    updated = parse_query_node(state)
    assert len(updated.errors) > 0
