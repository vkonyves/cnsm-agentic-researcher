from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Iterable

from .schemas import LiteratureRecord


def _request_json(
    url: str,
    *,
    user_agent: str,
    timeout: int = 60,
    attempts: int = 3,
) -> dict[str, Any]:
    """
    Execute an HTTP GET request and parse a JSON response.

    Retries transient network errors, HTTP 429 responses,
    and server-side failures. Invalid client requests fail
    immediately with the response body included.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
    )

    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                return json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as exc:
            last_error = exc

            response_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            # Do not include the request URL in the error,
            # because the OpenAlex API key is embedded in it.
            message = (
                f"Literature API request failed with "
                f"HTTP {exc.code}: "
                f"{response_body[:1000]}"
            )

            # Retry rate limiting and server errors.
            if (
                exc.code == 429
                or 500 <= exc.code < 600
            ):
                if attempt < attempts:
                    delay_seconds = (
                        5 * (2 ** (attempt - 1))
                    )

                    print(message)
                    print(
                        "Retrying literature request in "
                        f"{delay_seconds} seconds..."
                    )

                    time.sleep(delay_seconds)
                    continue

            raise RuntimeError(
                message
            ) from exc

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = exc

            if attempt < attempts:
                delay_seconds = (
                    5 * (2 ** (attempt - 1))
                )

                print(
                    "Literature request failed on "
                    f"attempt {attempt}/{attempts}: "
                    f"{exc}"
                )
                print(
                    "Retrying literature request in "
                    f"{delay_seconds} seconds..."
                )

                time.sleep(delay_seconds)
                continue

    raise RuntimeError(
        "Literature API request failed after "
        f"{attempts} attempts."
    ) from last_error


def _abstract(
    index: dict[str, list[int]] | None,
) -> str | None:
    """
    Reconstruct an OpenAlex abstract from its inverted index.
    """
    if not index:
        return None

    pairs = sorted(
        (
            offset,
            word,
        )
        for word, offsets in index.items()
        for offset in offsets
    )

    return " ".join(
        word
        for _, word in pairs
    )


def search_openalex(
    query: str,
    *,
    per_page: int = 25,
    mailto: str | None = None,
    api_key: str | None = None,
) -> list[LiteratureRecord]:
    """
    Search OpenAlex for scholarly works.
    """
    cleaned_query = " ".join(
        query.strip().split()
    )

    if not cleaned_query:
        return []

    effective_api_key = (
        api_key
        or os.getenv("OPENALEX_API_KEY")
    )

    if not effective_api_key:
        raise RuntimeError(
            "OPENALEX_API_KEY is not available. "
            "Add it to the repository .env file "
            "or export it in the shell."
        )

    effective_mailto = (
        mailto
        or os.getenv("OPENALEX_MAILTO")
    )

    params: dict[str, str] = {
        "search": cleaned_query,
        "per_page": str(
            min(
                max(per_page, 1),
                100,
            )
        ),
        "api_key": effective_api_key.strip(),
    }

    if effective_mailto:
        cleaned_mailto = (
            effective_mailto.strip()
        )

        if (
            "@" in cleaned_mailto
            and " " not in cleaned_mailto
        ):
            params["mailto"] = (
                cleaned_mailto
            )
        else:
            print(
                "Ignoring malformed "
                "OPENALEX_MAILTO value."
            )

    url = (
        "https://api.openalex.org/works?"
        + urllib.parse.urlencode(params)
    )

    print(
        "Searching OpenAlex:",
        repr(cleaned_query),
    )

    data = _request_json(
        url,
        user_agent=(
            "cnsm-agentic/0.8 "
            f"mailto:{effective_mailto or 'not-provided'}"
        ),
    )

    result: list[LiteratureRecord] = []

    for item in data.get(
        "results",
        [],
    ):
        doi = item.get("doi")

        if isinstance(doi, str):
            doi = doi.removeprefix(
                "https://doi.org/"
            )

        primary_location = (
            item.get("primary_location")
            or {}
        )

        result.append(
            LiteratureRecord(
                record_id=str(
                    item.get(
                        "id",
                        "",
                    )
                ),
                title=str(
                    item.get(
                        "title",
                        "",
                    )
                    or item.get(
                        "display_name",
                        "",
                    )
                ).strip(),
                abstract=_abstract(
                    item.get(
                        "abstract_inverted_index"
                    )
                ),
                publication_year=(
                    item.get(
                        "publication_year"
                    )
                ),
                doi=doi,
                url=(
                    primary_location.get(
                        "landing_page_url"
                    )
                ),
                source_api="openalex",
                authors=[
                    author_name
                    for authorship in item.get(
                        "authorships",
                        [],
                    )
                    if (
                        author_name := (
                            authorship.get(
                                "author",
                                {},
                            ).get(
                                "display_name"
                            )
                        )
                    )
                ],
                cited_by_count=(
                    item.get(
                        "cited_by_count"
                    )
                ),
                retrieved_for_queries=[
                    cleaned_query
                ],
            )
        )

    return result


def search_crossref(
    query: str,
    *,
    rows: int = 25,
    mailto: str | None = None,
) -> list[LiteratureRecord]:
    """
    Search Crossref for scholarly works.
    """
    cleaned_query = " ".join(
        query.strip().split()
    )

    if not cleaned_query:
        return []

    params = {
        "query.bibliographic": (
            cleaned_query
        ),
        "rows": str(
            max(
                rows,
                1,
            )
        ),
    }

    if mailto:
        cleaned_mailto = (
            mailto.strip()
        )

        if (
            "@" in cleaned_mailto
            and " " not in cleaned_mailto
        ):
            params["mailto"] = (
                cleaned_mailto
            )

    url = (
        "https://api.crossref.org/works?"
        + urllib.parse.urlencode(params)
    )

    print(
        "Searching Crossref:",
        repr(cleaned_query),
    )

    data = _request_json(
        url,
        user_agent=(
            "cnsm-agentic/0.8 "
            f"mailto:{mailto or 'not-provided'}"
        ),
    )

    result: list[LiteratureRecord] = []

    items = (
        data.get(
            "message",
            {},
        ).get(
            "items",
            [],
        )
    )

    for item in items:
        titles = (
            item.get("title")
            or []
        )

        date_parts = (
            item.get(
                "published",
                {},
            ).get(
                "date-parts"
            )
            or []
        )

        year = None

        if (
            date_parts
            and date_parts[0]
        ):
            try:
                year = int(
                    date_parts[0][0]
                )
            except (
                TypeError,
                ValueError,
            ):
                year = None

        doi = item.get("DOI")

        authors = []

        for author in item.get(
            "author",
            [],
        ):
            name = " ".join(
                value
                for value in (
                    author.get(
                        "given",
                        "",
                    ),
                    author.get(
                        "family",
                        "",
                    ),
                )
                if value
            )

            if name:
                authors.append(name)

        record_id = (
            f"doi:{doi}"
            if doi
            else str(
                item.get(
                    "URL",
                    titles[0]
                    if titles
                    else "",
                )
            )
        )

        result.append(
            LiteratureRecord(
                record_id=record_id,
                title=(
                    str(
                        titles[0]
                    ).strip()
                    if titles
                    else ""
                ),
                abstract=(
                    item.get(
                        "abstract"
                    )
                ),
                publication_year=year,
                doi=doi,
                url=item.get(
                    "URL"
                ),
                source_api="crossref",
                authors=authors,
                cited_by_count=(
                    item.get(
                        "is-referenced-by-count"
                    )
                ),
                retrieved_for_queries=[
                    cleaned_query
                ],
            )
        )

    return result


def deduplicate_records(
    records: Iterable[
        LiteratureRecord
    ],
) -> list[LiteratureRecord]:
    """
    Deduplicate records by DOI where available,
    otherwise by a normalized title.
    """
    chosen: dict[
        str,
        LiteratureRecord,
    ] = {}

    queries: defaultdict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in records:
        doi = (
            record.doi
            .strip()
            .lower()
            .removeprefix(
                "https://doi.org/"
            )
            if record.doi
            else None
        )

        title = " ".join(
            "".join(
                character.lower()
                if character.isalnum()
                else " "
                for character
                in record.title
            ).split()
        )

        if doi:
            key = f"doi:{doi}"
        elif title:
            key = f"title:{title}"
        else:
            key = (
                f"record:{record.source_api}:"
                f"{record.record_id}"
            )

        queries[key].update(
            record.retrieved_for_queries
        )

        if key not in chosen:
            chosen[key] = record

        elif (
            not chosen[key].abstract
            and record.abstract
        ):
            chosen[key].abstract = (
                record.abstract
            )

    for key, record in chosen.items():
        record.retrieved_for_queries = (
            sorted(
                queries[key]
            )
        )

    return sorted(
        chosen.values(),
        key=lambda record: (
            record.publication_year
            or 0,
            record.cited_by_count
            or 0,
        ),
        reverse=True,
    )


def discover_literature(
    queries: list[str],
    *,
    per_source_per_query: int = 25,
) -> list[LiteratureRecord]:
    """
    Query OpenAlex and Crossref for each planned query.

    Individual source-query failures are logged and do
    not terminate the run as long as at least one source
    returns records.
    """
    mailto = os.getenv(
        "OPENALEX_MAILTO"
    )

    openalex_api_key = os.getenv(
        "OPENALEX_API_KEY"
    )

    records: list[
        LiteratureRecord
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    for query in queries:
        try:
            records.extend(
                search_openalex(
                    query,
                    per_page=(
                        per_source_per_query
                    ),
                    mailto=mailto,
                    api_key=(
                        openalex_api_key
                    ),
                )
            )

        except Exception as exc:
            print(
                "OpenAlex search failed for "
                f"{query!r}: {exc}"
            )

            failures.append(
                {
                    "source": "openalex",
                    "query": query,
                    "error": str(exc),
                }
            )

        try:
            records.extend(
                search_crossref(
                    query,
                    rows=(
                        per_source_per_query
                    ),
                    mailto=mailto,
                )
            )

        except Exception as exc:
            print(
                "Crossref search failed for "
                f"{query!r}: {exc}"
            )

            failures.append(
                {
                    "source": "crossref",
                    "query": query,
                    "error": str(exc),
                }
            )

    if not records:
        raise RuntimeError(
            "All literature retrieval requests failed."
        )

    if failures:
        print(
            "Literature discovery completed with "
            f"{len(failures)} failed source-query "
            "requests."
        )

        for failure in failures:
            print(
                f"- {failure['source']} | "
                f"{failure['query']!r} | "
                f"{failure['error']}"
            )

    return deduplicate_records(
        records
    )