"""Tests for arXiv client & XML parsing.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import xml.etree.ElementTree as ET
from arxiv_digest.nodes.arxiv_client import parse_atom_entry

SAMPLE_ATOM_ENTRY = """
<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2024-01-23T18:00:00Z</published>
    <updated>2024-01-24T12:00:00Z</updated>
    <title> Fast and Efficient KV-Cache Compression </title>
    <summary> We propose an adaptive eviction technique for key-value caching in LLMs. </summary>
    <author>
        <name>Alice Smith</name>
    </author>
    <author>
        <name>Bob Jones</name>
    </author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
</entry>
"""


def test_parse_atom_entry():
    entry_el = ET.fromstring(SAMPLE_ATOM_ENTRY)
    paper = parse_atom_entry(entry_el)

    assert paper.arxiv_id == "2401.12345"
    assert paper.title == "Fast and Efficient KV-Cache Compression"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert "adaptive eviction technique" in paper.abstract
    assert paper.published_date == "2024-01-23"
    assert paper.primary_category == "cs.CL"
    assert "cs.AI" in paper.categories
