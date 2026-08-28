from cnsm_agentic.autonomous_research.literature import (
    _sanitize_openalex_query,
)


def test_openalex_wildcards_are_removed():
    query = (
        'LLM hallucinat* OR "incorrect" OR "unsafe" '
        '"network" OR "NetOps"'
    )

    cleaned = _sanitize_openalex_query(query)

    assert "*" not in cleaned
    assert "?" not in cleaned
    assert "hallucinat" in cleaned
    assert '"incorrect"' in cleaned
    assert '"NetOps"' in cleaned


def test_openalex_query_without_wildcards_is_unchanged():
    query = (
        'large language model OR LLM '
        '"network configuration"'
    )

    assert _sanitize_openalex_query(query) == query


def test_openalex_query_whitespace_is_normalized():
    query = '  LLM   NetOps   configuration  '

    assert (
        _sanitize_openalex_query(query)
        == "LLM NetOps configuration"
    )
