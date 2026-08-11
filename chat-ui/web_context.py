#!/usr/bin/env python3
"""Keyless web search + page text for chat-ui context injection.

Uses DuckDuckGo's HTML endpoint and plain HTTP fetches. No API keys.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
DDG_HTML = "https://html.duckduckgo.com/html/"
MAX_RESULTS_DEFAULT = 8
MAX_FETCH_DEFAULT = 5
MAX_SNIPPET_CHARS = 320
MAX_PAGE_CHARS = 1800
MAX_CONTEXT_CHARS = 16000
FETCH_TIMEOUT_S = 12
SEARCH_TIMEOUT_S = 20


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


class _DDGResultParser(HTMLParser):
    """Pull title/url/snippet triples from DuckDuckGo HTML results."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[SearchHit] = []
        self._in_result_a = False
        self._in_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: (v or "") for k, v in attrs}
        classes = set(ad.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._in_result_a = True
            self._title_parts = []
            self._href = ad.get("href") or None
        elif tag == "a" and "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []
        elif tag in {"td", "div"} and "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result_a:
            self._in_result_a = False
            title = _clean_text("".join(self._title_parts))
            url = _unwrap_ddg_url(self._href or "")
            if title and url and _http_url(url):
                self.hits.append(SearchHit(title=title, url=url, snippet=""))
            self._href = None
            self._title_parts = []
        elif tag in {"a", "td", "div"} and self._in_snippet:
            self._in_snippet = False
            snippet = _clean_text("".join(self._snippet_parts))
            if snippet and self.hits and not self.hits[-1].snippet:
                self.hits[-1].snippet = snippet[:MAX_SNIPPET_CHARS]
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_result_a:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)


class _TextExtractor(HTMLParser):
    """Rough visible-text extractor for fetched pages."""

    _SKIP = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if data.strip():
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _http_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _unwrap_ddg_url(href: str) -> str:
    if not href:
        return ""
    href = html.unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.netloc or "") and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg") or qs.get("u")
        if uddg:
            return unquote(uddg[0])
    if href.startswith("/"):
        return urljoin("https://duckduckgo.com", href)
    return href


def _http_get(url: str, timeout: float) -> tuple[str, str]:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        raw = resp.read(512 * 1024)
        charset = "utf-8"
        if "charset=" in ctype:
            charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        final = resp.geturl() or url
        return final, text


def search_duckduckgo(query: str, max_results: int = MAX_RESULTS_DEFAULT) -> list[SearchHit]:
    q = (query or "").strip()
    if not q:
        return []
    body = urlencode({"q": q}).encode("utf-8")
    req = Request(
        DDG_HTML,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html",
        },
        method="POST",
    )
    with urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
        html_text = resp.read().decode("utf-8", errors="replace")
    parser = _DDGResultParser()
    parser.feed(html_text)
    # Dedupe by URL, preserve order.
    seen: set[str] = set()
    out: list[SearchHit] = []
    for hit in parser.hits:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        out.append(hit)
        if len(out) >= max(1, max_results):
            break
    return out


def fetch_page_text(url: str, max_chars: int = MAX_PAGE_CHARS) -> dict[str, str]:
    if not _http_url(url):
        raise ValueError("url must be http(s)")
    final, html_text = _http_get(url, FETCH_TIMEOUT_S)
    # Prefer HTML extraction; fall back to plain text.
    ctype_hint = html_text.lstrip()[:40].lower()
    text = ""
    title = ""
    if "<html" in ctype_hint or "<!doctype" in ctype_hint or "<head" in html_text[:500].lower():
        extractor = _TextExtractor()
        try:
            extractor.feed(html_text)
            extractor.close()
            text = extractor.text()
            title = _clean_text(extractor.title)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html_text)
    else:
        text = html_text
    text = _clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return {"url": final, "title": title, "text": text[:max_chars]}


def _ordered_fetch_candidates(hits: list[SearchHit]) -> list[SearchHit]:
    """Prefer host diversity so one domain does not consume every fetch slot."""
    primary: list[SearchHit] = []
    secondary: list[SearchHit] = []
    seen_hosts: set[str] = set()
    for hit in hits:
        host = _host(hit.url)
        if host and host in seen_hosts:
            secondary.append(hit)
        else:
            if host:
                seen_hosts.add(host)
            primary.append(hit)
    return primary + secondary


def fetch_pages_until(
    hits: list[SearchHit],
    *,
    max_fetch: int,
    max_chars: int = MAX_PAGE_CHARS,
) -> tuple[list[dict[str, str]], list[str], int]:
    """Fetch until max_fetch successful pages, skipping failures and trying later hits."""
    want = max(0, max_fetch)
    if want == 0 or not hits:
        return [], [], 0
    pages: list[dict[str, str]] = []
    errors: list[str] = []
    seen_final: set[str] = set()
    attempted = 0
    for hit in _ordered_fetch_candidates(hits):
        if len(pages) >= want:
            break
        attempted += 1
        try:
            page = fetch_page_text(hit.url, max_chars=max_chars)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(f"{hit.url}: {exc}")
            continue
        text = (page.get("text") or "").strip()
        if not text:
            errors.append(f"{hit.url}: empty page text")
            continue
        final = page.get("url") or hit.url
        if final in seen_final:
            continue
        seen_final.add(final)
        if not page.get("title"):
            page["title"] = hit.title
        pages.append(page)
    return pages, errors, attempted


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def assemble_context(
    query: str,
    hits: list[SearchHit],
    pages: list[dict[str, str]],
    errors: list[str],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Build the prompt block, budgeting space so later pages are not wiped."""
    header = [
        "----- web context (DuckDuckGo HTML, no API key) -----",
        f"Query: {query}",
        "",
        "Search results:",
    ]
    if not hits:
        header.append("(no results)")
    for i, hit in enumerate(hits, 1):
        header.append(f"{i}. {hit.title}")
        header.append(f"   URL: {hit.url}")
        if hit.snippet:
            header.append(f"   {hit.snippet}")

    footer: list[str] = []
    if errors:
        footer.append("Fetch notes:")
        for err in errors[:8]:
            footer.append(f"- {err}")
    footer.append("----- end web context -----")

    head_text = "\n".join(header)
    foot_text = "\n".join(footer)
    reserved = len(head_text) + len(foot_text) + 64
    budget = max(0, max_chars - reserved)

    page_blocks: list[str] = []
    if pages and budget > 80:
        page_blocks.append("Fetched page text:")
        budget -= len(page_blocks[0]) + 1
        per = max(400, budget // max(1, len(pages)))
        for page in pages:
            if budget < 120:
                break
            title = page.get("title") or page.get("url") or "page"
            body = _clip(page.get("text") or "", min(MAX_PAGE_CHARS, per, budget - 80))
            block = f"### {title}\nURL: {page.get('url', '')}\n{body}\n"
            if len(block) > budget:
                block = _clip(block, budget)
            page_blocks.append(block.rstrip())
            budget -= len(block) + 1

    parts = [head_text]
    if page_blocks:
        parts.append("\n".join(page_blocks))
    parts.append(foot_text)
    context = "\n\n".join(parts).strip()
    if len(context) > max_chars:
        context = _clip(context, max_chars)
    return context


def build_web_context(
    query: str,
    *,
    max_results: int = MAX_RESULTS_DEFAULT,
    max_fetch: int = MAX_FETCH_DEFAULT,
    fetch_pages: bool = True,
) -> dict[str, Any]:
    """Search the web and optionally fetch top pages into a prompt block."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query required")

    try:
        hits = search_duckduckgo(q, max_results=max_results)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"web search failed: {exc}") from exc

    pages: list[dict[str, str]] = []
    errors: list[str] = []
    attempted = 0
    if fetch_pages:
        pages, errors, attempted = fetch_pages_until(hits, max_fetch=max_fetch)

    context = assemble_context(q, hits, pages, errors)

    return {
        "query": q,
        "results": [
            {"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits
        ],
        "pages_fetched": len(pages),
        "pages_attempted": attempted,
        "errors": errors,
        "context": context,
        "provider": "duckduckgo-html",
    }
