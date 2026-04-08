from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
from typing import Any

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
BIBLIO_FILE = BASE_DIR / "biblio.jsonl"
CORPUS_FILE = BASE_DIR / "corpus.jsonl"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    snippet_length: int = 200


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path.name}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    return ""


def pick_first(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in record:
            val = flatten_text(record.get(key)).strip()
            if val:
                return val
    return ""


def normalize_text(text: str) -> str:
    return text.lower().strip()


def make_snippet(text: str, query: str, width: int = 200) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    lower_text = text.lower()
    lower_query = query.lower()
    idx = lower_text.find(lower_query)

    if idx < 0:
        snippet = text[:width]
        return snippet + ("..." if len(text) > width else "")

    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(query) + width // 2)
    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def build_biblio_search_text(record: dict[str, Any]) -> str:
    # まずは軽い情報だけで絞る
    parts = [
        pick_first(record, ["Title", "title"]),
        pick_first(record, ["Citation", "citation"]),
        pick_first(record, ["Author", "author"]),
        pick_first(record, ["Journal", "journal"]),
        pick_first(record, ["Summary", "summary"]),
        pick_first(record, ["Tag", "tag", "Tags", "tags"]),
        pick_first(record, ["Text", "text"]),  # biblio に短い本文や要約がある場合用
    ]
    return " ".join(p for p in parts if p).strip()


def score_biblio_record(record: dict[str, Any], query: str) -> int:
    q = normalize_text(query)
    if not q:
        return 0

    title = normalize_text(pick_first(record, ["Title", "title"]))
    citation = normalize_text(pick_first(record, ["Citation", "citation"]))
    author = normalize_text(pick_first(record, ["Author", "author"]))
    journal = normalize_text(pick_first(record, ["Journal", "journal"]))
    summary = normalize_text(pick_first(record, ["Summary", "summary"]))
    tags = normalize_text(pick_first(record, ["Tag", "tag", "Tags", "tags"]))
    text = normalize_text(pick_first(record, ["Text", "text"]))

    # 単純な重みづけ
    score = 0
    score += title.count(q) * 10
    score += citation.count(q) * 6
    score += author.count(q) * 3
    score += journal.count(q) * 3
    score += summary.count(q) * 4
    score += tags.count(q) * 4
    score += text.count(q) * 2

    # 完全一致しない場合、語を分けて部分一致も少し見る
    if score == 0:
        terms = [t for t in query.strip().split() if t]
        search_text = " ".join([title, citation, author, journal, summary, tags, text])
        for term in terms:
            term_n = normalize_text(term)
            if not term_n:
                continue
            score += title.count(term_n) * 4
            score += citation.count(term_n) * 2
            score += summary.count(term_n) * 2
            score += text.count(term_n) * 1
            if term_n in search_text:
                score += 1

    return score


def build_corpus_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        key = pick_first(record, ["Key", "key"])
        if key:
            index[key] = record
    return index


def extract_corpus_text(record: dict[str, Any]) -> str:
    # 全文候補を優先順で拾う
    for key in ["Text", "text", "FullText", "fulltext", "Full_Text", "full_text", "Summary", "summary"]:
        val = flatten_text(record.get(key)).strip()
        if val:
            return val
    return flatten_text(record)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")

    top_k = max(1, min(req.top_k, 20))
    snippet_length = max(50, min(req.snippet_length, 1000))

    try:
        biblio_records = load_jsonl(BIBLIO_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"data file not found: {BIBLIO_FILE.name}")

    try:
        corpus_records = load_jsonl(CORPUS_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"data file not found: {CORPUS_FILE.name}")

    corpus_index = build_corpus_index(corpus_records)

    scored = []
    for i, record in enumerate(biblio_records):
        key = pick_first(record, ["Key", "key"])
        if not key:
            continue

        score = score_biblio_record(record, query)
        if score <= 0:
            continue

        scored.append({
            "rank_source_index": i,
            "score": score,
            "key": key,
            "biblio_record": record
        })

    scored.sort(key=lambda x: (-x["score"], x["rank_source_index"]))
    scored = scored[:top_k]

    results = []
    for item in scored:
        key = item["key"]
        biblio_record = item["biblio_record"]
        corpus_record = corpus_index.get(key, {})

        title = pick_first(biblio_record, ["Title", "title"]) or pick_first(corpus_record, ["Title", "title"])
        citation = pick_first(biblio_record, ["Citation", "citation"]) or pick_first(corpus_record, ["Citation", "citation"])
        author = pick_first(biblio_record, ["Author", "author"]) or pick_first(corpus_record, ["Author", "author"])
        year = pick_first(biblio_record, ["Year", "year"]) or pick_first(corpus_record, ["Year", "year"])

        corpus_text = extract_corpus_text(corpus_record)
        if not corpus_text:
            corpus_text = build_biblio_search_text(biblio_record)

        snippet = make_snippet(corpus_text, query, width=snippet_length)

        results.append({
            "Key": key,
            "score": item["score"],
            "title": title,
            "citation": citation,
            "author": author,
            "year": year,
            "snippet": snippet,
            "biblio_record": biblio_record,
            "corpus_record": corpus_record
        })

    return {
        "query": query,
        "count": len(results),
        "results": results
    }
