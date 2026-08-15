from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRecord:
    title: str
    url: str
    content: str
    source_type: str
    claim: str | None = None
    confidence: float = 0.75


DEFAULT_FIXTURES = [
    SourceRecord("LangGraph overview", "https://fixtures.local/langgraph", "LangGraph provides durable execution, checkpointing, and stateful orchestration for long-running agents.", "web", "LangGraph supports durable state and checkpointed agent execution.", 0.9),
    SourceRecord("Dynamic orchestration study", "https://arxiv.org/abs/2401.00001", "Dynamic delegation can adapt tool and specialist selection to each research question.", "academic", "Dynamic delegation changes the execution path based on the question.", 0.8),
    SourceRecord("Dynamic delegation overview", "https://fixtures.local/dynamic-delegation", "Dynamic delegation routes research tasks to specialized agents based on question and evidence needs.", "web", "Dynamic delegation routes tasks to specialists based on question and evidence needs.", 0.8),
    SourceRecord("Acme 2025 10-K", "https://sec.gov/fixtures/acme-10k", "Acme reported 2025 gross margin of 41%; higher freight costs reduced margin.", "regulatory", "Acme's 2025 gross margin declined because freight costs rose.", 0.95),
    SourceRecord("Acme investor update", "https://fixtures.local/acme-update", "Acme management said product mix, not freight costs, was the primary cause of the 2025 margin decline.", "web", "Acme's 2025 gross margin did not decline primarily because freight costs rose.", 0.7),
    SourceRecord("Python documentation", "https://fixtures.local/python", "Python 3.12 improves error messages and typing features.", "web", "Python 3.12 includes typing and diagnostic improvements.", 0.85),
]


def _terms(text: str) -> set[str]:
    stopwords = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "in", "is", "it", "no", "not", "of", "on", "or", "question", "the", "to", "was", "what", "with"}
    return set(re.findall(r"[a-z0-9]+", text.lower())) - stopwords


def _live_query(query: str) -> str:
    text = next((line.strip() for line in query.splitlines() if line.strip()), "")
    text = re.split(r"\b(?:address (?:(?:these|the) )?evidence gaps?|need evidence for|follow[- ]?up)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"^(?:research\s+question|question|task|research|investigate|find|search(?:\s+for)?|look\s+up|tell\s+me\s+about|please)\s*:?\s*", "", text, flags=re.IGNORECASE)
    stopwords = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does", "explain", "for", "from", "how", "in", "is", "it", "me", "more", "of", "on", "or", "please", "research", "should", "summary", "summarize", "tell", "that", "the", "these", "this", "to", "using", "what", "when", "where", "which", "who", "why", "will", "with"}
    words = (word for word in re.findall(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*", text) if word.lower() not in stopwords)
    return " ".join(dict.fromkeys(words))[:240].strip()


class FixtureSearchAdapter:
    def __init__(self, path: str | None = None) -> None:
        self.records = DEFAULT_FIXTURES if path is None else self._load(path)

    @staticmethod
    def _load(path: str) -> list[SourceRecord]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return [SourceRecord(**item) for item in data]
        except (OSError, ValueError, TypeError) as exc:
            raise ToolError(f"invalid fixture file: {exc}") from exc

    def search(self, query: str, source_types: set[str] | None = None, limit: int = 3) -> list[SourceRecord]:
        wanted = _terms(query)
        ranked = []
        for index, record in enumerate(self.records):
            if source_types and record.source_type not in source_types:
                continue
            score = len(wanted & _terms(f"{record.title} {record.content} {record.claim or ''}"))
            if score:
                ranked.append((-score, index, record))
        return [record for _, _, record in sorted(ranked)[:limit]]

    def fetch(self, url: str) -> SourceRecord:
        for record in self.records:
            if record.url == url:
                return record
        raise ToolError(f"fixture URL not found: {url}")


class TavilySearchAdapter:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def search(self, query: str, limit: int = 3) -> list[SourceRecord]:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise ToolError("live search requires TAVILY_API_KEY")
        endpoint = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")
        try:
            response = self.client.post(endpoint, json={"api_key": key, "query": _live_query(query), "max_results": limit, "search_depth": "basic"})
            response.raise_for_status()
            data = response.json()
            return [SourceRecord(item.get("title") or item["url"], item["url"], item.get("content", ""), "web") for item in data.get("results", [])[:limit]]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ToolError(f"Tavily search failed: {exc}") from exc


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


class ResearchTools:
    def __init__(self, mode: str, fixture_path: str | None = None, timeout: float = 10, max_bytes: int = 1_000_000) -> None:
        self.mode = mode
        self.max_bytes = max_bytes
        self.client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": os.getenv("SEC_USER_AGENT", "research-assistant/0.1 research@example.invalid")})
        self.fixtures = FixtureSearchAdapter(fixture_path)
        self.tavily = TavilySearchAdapter(self.client)

    def close(self) -> None:
        self.client.close()

    def invoke(self, name: str, query: str, documents: list[str]) -> list[SourceRecord]:
        methods = {
            "web_search": lambda: self.web_search(query),
            "fetch_url": lambda: self.fetch_url(query),
            "fetch_arxiv": lambda: self.fetch_arxiv(query),
            "fetch_sec": lambda: self.fetch_sec(query),
            "read_document": lambda: self.read_document(documents),
            "search_document": lambda: self.search_document(query, documents),
        }
        if name not in methods:
            raise ToolError(f"unknown tool: {name}")
        return methods[name]()

    def web_search(self, query: str) -> list[SourceRecord]:
        return self.fixtures.search(query, {"web"}) if self.mode == "fixture" else self.tavily.search(query)

    def fetch_url(self, url: str) -> list[SourceRecord]:
        if self.mode == "fixture":
            return [self.fixtures.fetch(url)]
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ToolError("fetch_url accepts only absolute HTTP(S) URLs")
        try:
            with self.client.stream("GET", url) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ToolError(f"response exceeds {self.max_bytes} bytes")
                    chunks.append(chunk)
                raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
            parser = _TextExtractor()
            parser.feed(raw)
            text = " ".join(parser.parts) if parser.parts else raw
            return [SourceRecord(url, str(response.url), text[:20_000], "web")]
        except httpx.HTTPError as exc:
            raise ToolError(f"URL fetch failed: {exc}") from exc

    def fetch_arxiv(self, query: str) -> list[SourceRecord]:
        if self.mode == "fixture":
            return self.fixtures.search(query, {"academic"})
        try:
            response = self.client.get("https://export.arxiv.org/api/query", params={"search_query": f"all:{_live_query(query)}", "start": 0, "max_results": 3})
            response.raise_for_status()
            root = ET.fromstring(response.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            return [SourceRecord(entry.findtext("a:title", "arXiv paper", ns).strip(), entry.findtext("a:id", "", ns), entry.findtext("a:summary", "", ns).strip(), "academic", confidence=0.85) for entry in root.findall("a:entry", ns)]
        except (httpx.HTTPError, ET.ParseError) as exc:
            raise ToolError(f"arXiv search failed: {exc}") from exc

    def fetch_sec(self, query: str) -> list[SourceRecord]:
        if self.mode == "fixture":
            return self.fixtures.search(query, {"regulatory"})
        try:
            response = self.client.get("https://www.sec.gov/files/company_tickers.json")
            response.raise_for_status()
            wanted = _terms(query)
            companies = list(response.json().values())
            corporate = {"co", "company", "corp", "corporation", "group", "holdings", "inc", "ltd", "plc"}
            ranked = []
            for item in companies:
                title_terms = _terms(item["title"]) - corporate
                score = (10 if item["ticker"].lower() in wanted else 0) + len(wanted & title_terms)
                if score:
                    ranked.append((score, item))
            match = max(ranked, key=lambda pair: pair[0])[1] if ranked else None
            if not match:
                return []
            cik = str(match["cik_str"]).zfill(10)
            filing_response = self.client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            filing_response.raise_for_status()
            recent = filing_response.json()["filings"]["recent"]
            records = []
            for index, form in enumerate(recent["form"]):
                if form not in {"10-K", "10-Q", "8-K"}:
                    continue
                accession = recent["accessionNumber"][index]
                primary = recent["primaryDocument"][index]
                url = f"https://www.sec.gov/Archives/edgar/data/{match['cik_str']}/{accession.replace('-', '')}/{primary}"
                content = f"{match['title']} filed {form} on {recent['filingDate'][index]} for period {recent['reportDate'][index]}."
                records.append(SourceRecord(f"{match['title']} {form}", url, content, "regulatory", confidence=0.95))
                if len(records) == 3:
                    break
            return records
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ToolError(f"SEC search failed: {exc}") from exc

    def read_document(self, documents: list[str]) -> list[SourceRecord]:
        return [self._read_one(path) for path in documents]

    def search_document(self, query: str, documents: list[str]) -> list[SourceRecord]:
        wanted = _terms(query)
        results = []
        for path in documents:
            record = self._read_one(path)
            passages = [part.strip() for part in re.split(r"\n\s*\n", record.content) if part.strip()]
            ranked = sorted(((len(wanted & _terms(part)), index, part) for index, part in enumerate(passages)), reverse=True)
            matches = [part for score, _, part in ranked if score][:3]
            if matches:
                results.append(SourceRecord(record.title, record.url, "\n\n".join(matches), "document", confidence=0.9))
        return results

    @staticmethod
    def _read_one(path: str) -> SourceRecord:
        file = Path(path).resolve()
        if file.suffix.lower() not in {".md", ".markdown", ".txt"}:
            raise ToolError(f"unsupported document type: {file.suffix or '<none>'}")
        try:
            if file.stat().st_size > 2_000_000:
                raise ToolError("document exceeds 2000000 bytes")
            content = file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"document read failed: {exc}") from exc
        return SourceRecord(file.name, file.as_uri(), content, "document", confidence=0.9)
