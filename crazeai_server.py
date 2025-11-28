import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================
# ENV VARIABLES
# =============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# =============================
# FASTAPI APP SETUP
# =============================
app = FastAPI(title="CrazeAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# MODELS
# =============================
class ChatRequest(BaseModel):
    message: str

class WebResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str

class ChatResponse(BaseModel):
    answer: str
    results: List[WebResult]

# =============================
# GOOGLE SEARCH
# =============================
async def search_google(query: str):
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "num": 3
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        data = r.json()

    results = []
    if "items" in data:
        for item in data["items"]:
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google"
                )
            )
    return results

# =============================
# DUCKDUCKGO
# =============================
async def search_duckduckgo(query: str):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        data = r.json()

    results = []
    abstract = data.get("AbstractText") or ""

    if abstract:
        results.append(
            WebResult(
                title=data.get("Heading") or "DuckDuckGo Answer",
                url=data.get("AbstractURL") or "",
                snippet=abstract,
                source="duckduckgo"
            )
        )
    return results

# =============================
# WIKIPEDIA
# =============================
async def search_wikipedia(query: str):
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(search_url, params=params)
        data = r.json()

    results = []
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return results

    first = search_results[0]
    title = first["title"]
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    summary_url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
        f"{title.replace(' ', '_')}"
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        r2 = await client.get(summary_url)
        summary_data = r2.json()

    snippet = summary_data.get("extract") or first.get("snippet", "")

    results.append(
        WebResult(
            title=title,
            url=page_url,
            snippet=snippet,
            source="wikipedia"
        )
    )

    return results

# =============================
# AGGREGATE SEARCH (Google + DDG + Wiki)
# =============================
async def aggregate_search(query: str):
    google_res = await search_google(query)
    ddg_res = await search_duckduckgo(query)
    wiki_res = await search_wikipedia(query)

    all_results = google_res + ddg_res + wiki_res

    # remove duplicates
    seen = set()
    unique = []

    for r in all_results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)

    return unique[:7]

# =============================
# ANSWER BUILDER
# =============================
def build_answer(question: str, results: List[WebResult]):
    if not results:
        return f"I couldn't find anything for: {question}"

    lines = [f"Here's what I found about **{question}**:\n"]

    for r in results[:3]:
        lines.append(f"- {r.snippet} (source: {r.source})")

    lines.append("\nOpen the links below for full details.")

    return "\n".join(lines)

# =============================
# MAIN ENDPOINT
# =============================
@app.post("/api/crazeai", response_model=ChatResponse)
async def crazeai_chat(req: ChatRequest):
    q = req.message
    results = await aggregate_search(q)
    answer = build_answer(q, results)
    return ChatResponse(answer=answer, results=results)

# =============================
# LOCAL RUN
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("crazeai_server:app", host="0.0.0.0", port=8000, reload=True)
