from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
from typing import Any

app = FastAPI()

# リポジトリ直下に置いてある前提
DATA_FILE = Path(__file__).resolve().parent / "MyPublications.jsonl"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                # 壊れた1行があっても全体停止しない
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


def record_text(record: dict[str, Any]) -> str:
    return flatten_text(record)


def pick_first(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in record:
            val = flatten_text(record.get(key)).strip()
            if val:
                return val
    return ""


def make_snippet(text: str, query: str, width: int = 160) -> str:
    if not text:
        return ""
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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")

    top_k = max(1, min(req.top_k, 20))

    try:
        records = load_jsonl(DATA_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"data file not found: {DATA_FILE.name}")

    results = []
    q_lower = query.lower()

    for i, record in enumerate(records):
        text = record_text(record)
        if not text:
            continue

        text_lower = text.lower()
        score = text_lower.count(q_lower)
        if score <= 0:
            continue

        title = pick_first(record, ["Title", "title"])
        citation = pick_first(record, ["Citation", "citation"])
        snippet = make_snippet(text, query)

        results.append({
            "rank_source_index": i,
            "score": score,
            "title": title,
            "citation": citation,
            "snippet": snippet,
            "record": record
        })

    results.sort(key=lambda x: (-x["score"], x["rank_source_index"]))

    return {
        "query": query,
        "count": len(results[:top_k]),
        "results": results[:top_k]
    }
