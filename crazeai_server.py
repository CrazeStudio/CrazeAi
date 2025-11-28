import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ========== CONFIG ==========
# Put your keys into a .env file or env variables
BING_API_KEY = os.getenv("BING_API_KEY")        # optional but recommended

# ========== FASTAPI APP ==========
app = FastAPI(title="CrazeAi API")

# allow frontend from anywhere (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODELS ----------
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


# ========== SEARCH PROVIDERS ==========

async def search_bing(query: str) -> List[WebResult]:
    if not BING_API_KEY:
        return []

    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": 3}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=headers, params=params)
        data = r.json()

    results: List[WebResult] = []
    if "webPages" in data:
        for item in data["webPages"]["value"]:
            results.append(
                WebResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="bing",
                )
            )
    return results


async def search_duckduckgo(query: str) -> List[WebResult]:
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        data = r.json()

    results: List[WebResult] = []

    # main abstract
    abstract = data.get("AbstractText") or ""
    if abstract:
        results.append(
            WebResult(
                title=data.get("Heading") or "DuckDuckGo Instant Answer",
                url=data.get("AbstractURL") or "",
                snippet=abstract,
                source="duckduckgo",
            )
        )

    # related topics
    for topic in data.get("RelatedTopics", [])[:2]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append(
                WebResult(
                    title=topic.get("Text")[:80],
                    url=topic.get("FirstURL", ""),
                    snippet=topic.get("Text", ""),
                    source="duckduckgo",
                )
            )

    return results


async def search_wikipedia(query: str) -> List[WebResult]:
    # simple search -> first result summary
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(search_url, params=params)
        data = r.json()

    results: List[WebResult] = []
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return results

    first = search_results[0]
    title = first["title"]
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    # second call: summary
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
            source="wikipedia",
        )
    )
    return results


async def aggregate_search(query: str) -> List[WebResult]:
    """Call multiple systems and merge results."""
    bing_results: List[WebResult] = []
    ddg_results: List[WebResult] = []
    wiki_results: List[WebResult] = []

    # call in parallel
    async with httpx.AsyncClient(timeout=10.0) as client:
        # we can't share client easily with the functions above,
        # so just call them normally (simple way for you)
        pass

    # Simpler: just call sequentially (still fine for now)
    bing_results = await search_bing(query)
    ddg_results = await search_duckduckgo(query)
    wiki_results = await search_wikipedia(query)

    all_results = bing_results + ddg_results + wiki_results

    # remove duplicates by url
    seen = set()
    unique: List[WebResult] = []
    for r in all_results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            unique.append(r)

    # limit total
    return unique[:7]


# ========== SIMPLE ANSWER BUILDER ==========

def build_answer(question: str, results: List[WebResult]) -> str:
    if not results:
        return (
            "I couldn't find anything clear on the web right now. "
            "Try rephrasing your question or being more specific."
        )

    # Take top few snippets and stitch a human-style answer
    lines = [f"Here's what I found for: **{question}**\n"]

    for r in results[:3]:
        lines.append(f"- {r.snippet.strip()} (source: {r.source})")

    lines.append(
        "\nThese are web results. Always double-check important info "
        "by opening the original links."
    )

    return "\n".join(lines)


# ========== MAIN ENDPOINT ==========

@app.post("/api/crazeai", response_model=ChatResponse)
async def crazeai_chat(body: ChatRequest):
    query = body.message.strip()
    results = await aggregate_search(query)
    answer = build_answer(query, results)
    return ChatResponse(answer=answer, results=results)


# for local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("crazeai_server:app", host="0.0.0.0", port=8000, reload=True)
