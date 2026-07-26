from cnsm_agentic.acquisition import (
    classify_asset,
    is_direct_download_url,
    probable_filename,
)


def test_direct_zip_url() -> None:
    url = (
        "https://github.com/example/project/raw/main/"
        "data/benchmark.zip"
    )

    assert is_direct_download_url(url)
    assert probable_filename(url) == "benchmark.zip"


def test_direct_json_url() -> None:
    url = (
        "https://raw.githubusercontent.com/example/project/"
        "main/data/answers.json"
    )

    asset = classify_asset(
        resource_name="ExampleBench",
        source_url=url,
    )

    assert asset.action == "download"
    assert asset.expected_filename == "answers.json"


def test_doi_requires_verification() -> None:
    url = "https://doi.org/10.1234/example"

    asset = classify_asset(
        resource_name="ExampleBench",
        source_url=url,
    )

    assert asset.action == "manual_verification_required"


def test_repository_page_requires_verification() -> None:
    url = "https://github.com/example/project"

    asset = classify_asset(
        resource_name="ExampleBench",
        source_url=url,
    )

    assert asset.action == "manual_verification_required"
    
    
from cnsm_agentic.acquisition import (
    safe_filename,
    validate_download_url,
)


def test_safe_filename() -> None:
    assert (
        safe_filename("6G Bench/data")
        == "6G-Bench-data"
    )


def test_allow_listed_download_url() -> None:
    allowed, reason = validate_download_url(
        "https://raw.githubusercontent.com/"
        "example/project/main/data.json"
    )

    assert allowed
    assert reason is None


def test_reject_non_https_url() -> None:
    allowed, reason = validate_download_url(
        "http://raw.githubusercontent.com/"
        "example/project/main/data.json"
    )

    assert not allowed
    assert reason is not None


def test_reject_unknown_host() -> None:
    allowed, reason = validate_download_url(
        "https://example.com/data.json"
    )

    assert not allowed
    assert reason is not None    
