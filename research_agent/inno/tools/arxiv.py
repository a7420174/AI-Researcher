# arxiv_tools.py
from __future__ import annotations

import os
import re
import time
import tarfile
import urllib.parse
from typing import Any, Dict, List, Optional

import feedparser
import requests

try:
    import arxiv  # optional (used by get_arxiv_paper_meta)
except Exception:
    arxiv = None  # handled gracefully at runtime

from research_agent.inno.registry import register_tool


# ---------- 내부 유틸 ----------
def _sanitize_filename(name: str) -> str:
    """Make a filesystem-friendly filename."""
    name = re.sub(r"[^\w\-\. ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name.lower() or "untitled"


def _resolve_paths(
    title: str,
    local_root: Optional[str],
    workplace_name: Optional[str],
    env: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Resolve output directories for source (.tar.gz) and extracted .tex file.
    Priority:
      1) Provided local_root + workplace_name
      2) If env provides a base working dir (optional)
      3) Current working directory
    """
    safe = _sanitize_filename(title)

    # Base directory
    if local_root and workplace_name:
        base = os.path.join(local_root, workplace_name)
    elif env is not None and hasattr(env, "working_dir") and env.working_dir:
        base = os.path.join(str(env.working_dir))
    else:
        base = os.getcwd()

    src_dir = os.path.join(base, "paper_source") if (local_root and workplace_name) else os.path.join(base, "arxiv_source")
    tex_dir = os.path.join(base, "papers") if (local_root and workplace_name) else os.path.join(base, "arxiv_papers")

    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)

    return {
        "src_dir": src_dir,
        "tex_dir": tex_dir,
        "src_path": os.path.join(src_dir, f"{safe}.tar.gz"),
        "tex_path": os.path.join(tex_dir, f"{safe}.tex"),
        # For user message when local_root/workplace_name form is used:
        "display_tex_path": (f"/{os.path.basename(base)}/papers/{safe}.tex")
        if (local_root and workplace_name)
        else os.path.relpath(os.path.join(tex_dir, f"{safe}.tex"), start=base),
    }


# ---------- 툴: arXiv 검색 ----------
@register_tool("search_arxiv")
def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search arXiv papers by title keywords and return a list of results.

    Args:
        query: Search keywords (title-focused).
        max_results: Maximum number of results.

    Returns:
        A list of dicts: {title, authors, published, summary, url, pdf_url}
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_query = urllib.parse.quote(query)

    params = {
        "search_query": f"ti:{search_query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    query_url = base_url + urllib.parse.urlencode(params)
    feed = feedparser.parse(query_url)

    papers: List[Dict[str, Any]] = []
    for entry in feed.entries:
        paper = {
            "title": entry.title,
            "authors": [a.name for a in getattr(entry, "authors", [])],
            "published": getattr(entry, "published", None),
            "summary": getattr(entry, "summary", None),
            "url": entry.link,
            "pdf_url": next((l.href for l in entry.links if getattr(l, "type", "") == "application/pdf"), None),
        }
        papers.append(paper)
        # be polite to arXiv API
        time.sleep(0.5)

    return papers


# ---------- 툴: 메타데이터 조회 ----------
@register_tool("get_arxiv_paper_meta")
def get_arxiv_paper_meta(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve arXiv paper metadata by ID or URL.

    Args:
        arxiv_id: Full URL or bare ID ('2305.02759' or '2305.02759v4').

    Returns:
        Dict of metadata, or None if retrieval fails or arxiv lib is unavailable.
    """
    if arxiv is None:
        return None

    try:
        # Extract ID if URL provided
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]
        base_id = arxiv_id.split("v")[0]

        client = arxiv.Client()
        search = arxiv.Search(id_list=[base_id], max_results=1)
        paper = next(client.results(search))

        meta = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "categories": paper.categories,
            "published": paper.published,
            "updated": paper.updated,
            "doi": paper.doi,
            "pdf_url": paper.pdf_url,
            "primary_category": paper.primary_category,
            "comment": paper.comment,
            "journal_ref": paper.journal_ref,
            "version": paper.entry_id.split("v")[-1] if "v" in paper.entry_id else "1",
        }
        return meta
    except Exception as e:
        print(f"[get_arxiv_paper_meta] error: {e}")
        return None


# ---------- 툴: TAR.GZ에서 .tex 추출/병합 ----------
@register_tool("extract_tex_content")
def extract_tex_content(tar_path: str) -> str:
    """
    Extract and concatenate all .tex file contents from a tar.gz archive.

    Args:
        tar_path: Path to tar.gz file.

    Returns:
        Combined string of filename headers + content for all .tex files.
    """
    try:
        chunks: List[str] = []
        with tarfile.open(tar_path, "r:gz") as tar:
            tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            for member in tex_files:
                f = tar.extractfile(member)
                if f is None:
                    continue
                # Robust decoding
                try:
                    content = f.read().decode("utf-8")
                except UnicodeDecodeError:
                    f.seek(0)
                    content = f.read().decode("latin-1")
                header = f"\n{'='*50}\nFilename: {member.name}\n{'='*50}\n"
                chunks.append(header)
                chunks.append(content)
                chunks.append("\n\n")
        return "".join(chunks)
    except Exception as e:
        return f"Extract failed with error: {str(e)}"


# ---------- 툴: 단일 논문 소스 다운로드 ----------
@register_tool("download_arxiv_source")
def download_arxiv_source(
    arxiv_url: str,
    local_root: Optional[str] = None,
    workplace_name: Optional[str] = None,
    title: str = "",
    *,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Download arXiv source (.tar.gz) for a single paper and extract .tex contents.

    Args:
        arxiv_url: e.g. 'http://arxiv.org/abs/2006.11239v2'
        local_root: base local directory (optional)
        workplace_name: workspace name under local_root (optional)
        title: paper title (used for filenames)
        env: optional environment (e.g., DockerEnv) – if provided, may supply a base working dir.

    Returns:
        dict: {"status": 0/-1, "message": str, "path": <tex_path or None>}
    """
    try:
        # 1) arXiv ID from URL
        m = re.search(r"abs/([^/\s]+)", arxiv_url)
        if not m:
            return {"status": -1, "message": "Invalid arXiv URL format.", "path": None}
        paper_id = m.group(1)

        # 2) Build source URL and request
        source_url = f"http://arxiv.org/src/{paper_id}"
        resp = requests.get(source_url)
        if resp.status_code != 200:
            return {
                "status": -1,
                "message": f"Download failed with HTTP status code {resp.status_code}",
                "path": None,
            }

        # 3) Resolve paths and save .tar.gz
        title_for_name = title or paper_id
        paths = _resolve_paths(title_for_name, local_root, workplace_name, env)
        with open(paths["src_path"], "wb") as f:
            f.write(resp.content)

        # 4) Extract .tex content and save as a single .tex
        tex_content = extract_tex_content(paths["src_path"])
        with open(paths["tex_path"], "w", encoding="utf-8") as f:
            f.write(tex_content)

        return {
            "status": 0,
            "message": f"Download and extract succeeded for '{title_for_name}'",
            "path": paths["display_tex_path"],
        }
    except Exception as e:
        return {"status": -1, "message": f"Download failed with error: {str(e)}", "path": None}


# ---------- 툴: 제목 목록 일괄 다운로드 ----------
@register_tool("download_arxiv_source_by_title")
def download_arxiv_source_by_title(
    paper_list: List[str],
    local_root: Optional[str] = None,
    workplace_name: Optional[str] = None,
    *,
    env: Optional[Any] = None,
) -> str:
    """
    For each title in `paper_list`, search the most relevant arXiv entry and download its source.

    Args:
        paper_list: list of titles
        local_root: optional base directory
        workplace_name: optional workspace under base
        env: optional environment (e.g., DockerEnv)

    Returns:
        Multi-line status message for all titles.
    """
    messages: List[str] = []
    for title in paper_list:
        results = search_arxiv(title, max_results=1)
        if not results:
            messages.append(f"Cannot find the paper '{title}' in arXiv")
            continue

        url = results[0].get("url")
        info = download_arxiv_source(url, local_root, workplace_name, title=title, env=env)
        if info["status"] == -1:
            messages.append(info["message"])
        else:
            messages.append(info["message"] + f"\nSaved to: {info['path']}")
    return "\n".join(messages)

# ---------- 툴: 단일 제목 편의 다운로드 ----------
@register_tool("download_arxiv_by_title")
def download_arxiv_by_title(title: str, *, env: Optional[Any] = None) -> str:
    """
    Convenience wrapper: search a title and download its source into default folders.

    Args:
        title: paper title
        env: optional environment for base path

    Returns:
        Status string.
    """
    results = search_arxiv(title, max_results=1)
    if not results:
        return f"Cannot find the paper '{title}' in arXiv"

    info = download_arxiv_source(results[0]["url"], title=title, env=env)
    return info["message"] + (f"\nSaved to: {info['path']}" if info["path"] else "")
