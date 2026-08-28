from cnsm_agentic.autonomous_research.publication_renderer import (
    _latex_escape,
)


MASTER_PROMPT_SHA = (
    "1872df1e1805d2d96940456ca016bd66"
    "5d1d5196add77f5acdf1582bb39b15ba"
)


def test_sha256_gets_safe_latex_breakpoints():
    rendered = _latex_escape(
        "Immutable SHA-256 = "
        + MASTER_PROMPT_SHA
        + "."
    )

    assert MASTER_PROMPT_SHA not in rendered

    reconstructed = rendered.replace(
        r"\allowbreak{}",
        "",
    )

    assert MASTER_PROMPT_SHA in reconstructed
    assert rendered.count(r"\allowbreak{}") == 7


def test_sha256_breaking_preserves_every_digest_character():
    rendered = _latex_escape(MASTER_PROMPT_SHA)

    reconstructed = rendered.replace(
        r"\allowbreak{}",
        "",
    )

    assert reconstructed == MASTER_PROMPT_SHA


def test_short_hexadecimal_tokens_are_not_modified():
    token = "1872df1e1805d2d9"

    assert _latex_escape(token) == token
