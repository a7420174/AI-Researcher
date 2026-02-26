# openalex_tools.py
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

from research_agent.inno.registry import register_tool

OPENALEX_BASE = "https://api.openalex.org"

DEFAULT_SELECT = (
    "id,doi,title,primary_location,publication_year,abstract_inverted_index"
)


class OpenAlexAPIError(RuntimeError):
    pass


def _openalex_get(
    path: str,
    params: Dict[str, Any],
    *,
    user_agent: str,
    timeout_s: int = 30,
    max_retries: int = 5,
) -> Dict[str, Any]:
    url = f"{OPENALEX_BASE}{path}"
    headers = {"Accept": "application/json", "User-Agent": user_agent}

    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)

        if 200 <= resp.status_code < 300:
            return resp.json()

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt == max_retries:
                raise OpenAlexAPIError(
                    f"OpenAlex API failed: {resp.status_code} {resp.text}"
                )
            time.sleep((2**attempt) + random.random())
            continue

        raise OpenAlexAPIError(f"OpenAlex API error: {resp.status_code} {resp.text}")

    raise OpenAlexAPIError("Unexpected retry loop exit")


def _abstract_inverted_index_to_text(
    aii: Optional[Dict[str, List[int]]],
) -> Optional[str]:
    if not aii:
        return None
    pos_to_word: Dict[int, str] = {}
    for word, positions in aii.items():
        for p in positions:
            pos_to_word[p] = word
    return " ".join(pos_to_word[i] for i in sorted(pos_to_word.keys()))


def _merge_filters(existing: Optional[str], extra: str) -> str:
    if existing and existing.strip():
        return f"{existing},{extra}"
    return extra


@register_tool("openalex_search")
def openalex_search(
    query: Optional[str] = None,
    filter: Optional[str] = None,
    select: str = DEFAULT_SELECT,
    sort: Optional[str] = None,
    per_page: int = 50,
    page: int = 1,
    cursor: Optional[str] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    max_items: int = 50,
    abstract_as_text: bool = True,
) -> Dict[str, Any]:
    """
    OpenAlex를 이용한 학술 논문 검색 도구.

    Args:
        query: 검색어 (title과 abstract에서 검색)
        filter: 필터 (예: "publication_year:2024" 또는 "host_venue.name:Nature")
        select: 반환할 필드
        sort: 정렬 (예: "publication_year:desc" 또는 "cited_by_count:desc")
        per_page: 페이지당 결과 수 (최대 200)
        page: 페이지 번호
        cursor: 커서 페이지네이션용 커서
        sample: 무작위 샘플링 수
        seed: 샘플링 시드
        api_key: OpenAlex API 키 (선택)
        max_items: 최대 반환 아이템 수
        abstract_as_text: abstract를 텍스트로 변환

    Returns:
        검색 결과 (items, meta 포함)
    """
    per_page = max(1, min(int(per_page), 200))

    params: Dict[str, Any] = {"per-page": per_page}

    if query:
        ta_filter = f"title_and_abstract.search:{query}"
        filter = _merge_filters(filter, ta_filter)
    if filter:
        params["filter"] = filter
    if sort:
        params["sort"] = sort
    if select:
        params["select"] = select

    if sample is not None:
        params["sample"] = int(sample)
        if seed is not None:
            params["seed"] = int(seed)
        if cursor is not None:
            raise ValueError("Sampling does not support cursor pagination.")
        if page > 1 and seed is None:
            raise ValueError("When sampling beyond page 1, you must provide a seed.")
        params["page"] = int(page)
    else:
        if cursor:
            params["cursor"] = cursor
        else:
            params["page"] = int(page)

    if api_key:
        params["api_key"] = api_key

    user_agent = "research-agent-openalex/1.0"

    raw = _openalex_get("/works", params, user_agent=user_agent)

    meta = raw.get("meta", {}) or {}
    results = raw.get("results", []) or []

    items: List[Dict[str, Any]] = []
    for w in results[:max_items]:
        primary_location = w.get("primary_location") or {}
        primary_source_name = primary_location.get("raw_source_name")

        aii = w.get("abstract_inverted_index")
        abstract_text = (
            _abstract_inverted_index_to_text(aii) if abstract_as_text else None
        )

        items.append(
            {
                "id": w.get("id"),
                "doi": w.get("doi"),
                "title": w.get("title"),
                "publication_year": w.get("publication_year"),
                "primary_source_name": primary_source_name,
                "abstract": abstract_text,
            }
        )

    return {
        "meta": {
            "count": meta.get("count"),
            "page": meta.get("page"),
            "per_page": meta.get("per_page"),
            "next_cursor": meta.get("next_cursor"),
        },
        "items": items,
    }


@register_tool("openalex_search_by_doi")
def openalex_search_by_doi(
    dois: List[str],
    select: str = DEFAULT_SELECT,
    per_page: int = 50,
    api_key: Optional[str] = None,
    abstract_as_text: bool = True,
) -> Dict[str, Any]:
    """
    DOI 목록으로 학술 논문 검색.

    Args:
        dois: DOI 목록 (최대 50개)
        select: 반환할 필드
        per_page: 페이지당 결과 수
        api_key: OpenAlex API 키
        abstract_as_text: abstract를 텍스트로 변환

    Returns:
        검색 결과
    """
    if not dois:
        return {"meta": {}, "items": []}

    dois = dois[:50]
    filter_value = "|".join(dois)
    filter_str = f"doi:{filter_value}"

    return openalex_search(
        query=None,
        filter=filter_str,
        select=select,
        sort=None,
        per_page=min(per_page, 200),
        page=1,
        cursor=None,
        sample=None,
        seed=None,
        api_key=api_key,
        max_items=50,
        abstract_as_text=abstract_as_text,
    )


@register_tool("openalex_search_papers")
def openalex_search_papers(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    primary_source: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """
    학술 논문 검색 (단순화된 인터페이스).

    Args:
        query: 검색어
        year_from: 시작 연도
        year_to: 종료 연도
        primary_source: 주요 출처 (예: "Nature", "Science", "Cell")
        max_results: 최대 결과 수

    Returns:
        포맷된 검색 결과 문자열
    """
    filters_parts = []
    if year_from:
        filters_parts.append(f"publication_year:>={year_from}")
    if year_to:
        filters_parts.append(f"publication_year:<={year_to}")
    if primary_source:
        filters_parts.append(f"host_venue.display_name:{primary_source}")

    filter_str = ",".join(filters_parts) if filters_parts else None

    result = openalex_search(
        query=query,
        filter=filter_str,
        sort="publication_year:desc",
        per_page=min(max_results, 200),
        max_items=max_results,
    )

    items = result.get("items", [])
    if not items:
        return f"No results found for query: {query}"

    output_lines = [f"# OpenAlex Search Results for: {query}"]
    if year_from or year_to:
        output_lines.append(
            f"# Year range: {year_from or 'earliest'} - {year_to or 'latest'}"
        )
    output_lines.append("")

    for i, item in enumerate(items, 1):
        title = item.get("title", "No title")
        doi = item.get("doi", "No DOI")
        year = item.get("publication_year", "N/A")
        source = item.get("primary_source_name", "Unknown source")
        abstract = item.get("abstract", "")

        output_lines.append(f"## {i}. {title}")
        output_lines.append(f"- **Year:** {year}")
        output_lines.append(f"- **Source:** {source}")
        output_lines.append(f"- **DOI:** {doi}")
        if abstract:
            abstract_preview = (
                abstract[:500] + "..." if len(abstract) > 500 else abstract
            )
            output_lines.append(f"- **Abstract:** {abstract_preview}")
        output_lines.append("")

    output_lines.append(f"---")
    output_lines.append(
        f"Total results: {result.get('meta', {}).get('count', len(items))}"
    )

    return "\n".join(output_lines)
