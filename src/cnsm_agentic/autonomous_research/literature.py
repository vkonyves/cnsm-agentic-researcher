from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Iterable

from .schemas import LiteratureRecord


def _request_json(url: str, *, user_agent: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    pairs = sorted((offset, word) for word, offsets in index.items() for offset in offsets)
    return " ".join(word for _, word in pairs)


def search_openalex(query: str, *, per_page: int = 25, mailto: str | None = None) -> list[LiteratureRecord]:
    params = {"search": query, "per-page": str(per_page), "sort": "relevance_score:desc"}
    if mailto:
        params["mailto"] = mailto
    data = _request_json(
        "https://api.openalex.org/works?" + urllib.parse.urlencode(params),
        user_agent=f"cnsm-agentic/0.6 mailto:{mailto or 'not-provided'}",
    )
    result = []
    for item in data.get("results", []):
        doi = item.get("doi")
        if isinstance(doi, str):
            doi = doi.removeprefix("https://doi.org/")
        result.append(LiteratureRecord(
            record_id=str(item.get("id", "")),
            title=str(item.get("title", "")).strip(),
            abstract=_abstract(item.get("abstract_inverted_index")),
            publication_year=item.get("publication_year"),
            doi=doi,
            url=item.get("primary_location", {}).get("landing_page_url"),
            source_api="openalex",
            authors=[a.get("author", {}).get("display_name", "") for a in item.get("authorships", []) if a.get("author", {}).get("display_name")],
            cited_by_count=item.get("cited_by_count"),
            retrieved_for_queries=[query],
        ))
    return result


def search_crossref(query: str, *, rows: int = 25, mailto: str | None = None) -> list[LiteratureRecord]:
    params = {"query.bibliographic": query, "rows": str(rows)}
    if mailto:
        params["mailto"] = mailto
    data = _request_json(
        "https://api.crossref.org/works?" + urllib.parse.urlencode(params),
        user_agent=f"cnsm-agentic/0.6 mailto:{mailto or 'not-provided'}",
    )
    result = []
    for item in data.get("message", {}).get("items", []):
        titles = item.get("title") or []
        date_parts = item.get("published", {}).get("date-parts") or []
        year = int(date_parts[0][0]) if date_parts and date_parts[0] else None
        doi = item.get("DOI")
        result.append(LiteratureRecord(
            record_id=f"doi:{doi}" if doi else str(item.get("URL", titles[0] if titles else "")),
            title=str(titles[0]).strip() if titles else "",
            abstract=item.get("abstract"),
            publication_year=year,
            doi=doi,
            url=item.get("URL"),
            source_api="crossref",
            authors=[" ".join(v for v in [a.get("given", ""), a.get("family", "")] if v) for a in item.get("author", [])],
            cited_by_count=item.get("is-referenced-by-count"),
            retrieved_for_queries=[query],
        ))
    return result


def deduplicate_records(records: Iterable[LiteratureRecord]) -> list[LiteratureRecord]:
    chosen: dict[str, LiteratureRecord] = {}
    queries: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        doi = record.doi.strip().lower().removeprefix("https://doi.org/") if record.doi else None
        title = " ".join("".join(c.lower() if c.isalnum() else " " for c in record.title).split())
        key = f"doi:{doi}" if doi else f"title:{title}"
        queries[key].update(record.retrieved_for_queries)
        if key not in chosen:
            chosen[key] = record
        elif not chosen[key].abstract and record.abstract:
            chosen[key].abstract = record.abstract
    for key, record in chosen.items():
        record.retrieved_for_queries = sorted(queries[key])
    return sorted(chosen.values(), key=lambda r: (r.publication_year or 0, r.cited_by_count or 0), reverse=True)


def discover_literature(queries: list[str], *, per_source_per_query: int = 25) -> list[LiteratureRecord]:
    mailto = os.environ.get("OPENALEX_MAILTO")
    records = []
    for query in queries:
        records.extend(search_openalex(query, per_page=per_source_per_query, mailto=mailto))
        records.extend(search_crossref(query, rows=per_source_per_query, mailto=mailto))
    return deduplicate_records(records)
