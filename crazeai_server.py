import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

# ======================================
# ENV VARIABLES (GOOGLE SEARCH)
# ======================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# ======================================
# FASTAPI APP
# ======================================
app = FastAPI(title="CrazeAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# MODELS
# ======================================
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

# ======================================
# GOOGLE SEARCH
# ======================================
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

    async with httpx.AsyncClient(timeout=10) as client:
        data = (await client.get(url, params=params)).json()

    results = []
    for item in data.get("items", []):
        results.append(
            WebResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google"
            )
        )
    return results

# ======================================
# DUCKDUCKGO
# ======================================
async def search_duckduckgo(query: str):
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1}

    async with httpx.AsyncClient(timeout=10) as client:
        data = (await client.get(url, params=params)).json()

    results = []
    if data.get("AbstractText"):
        results.append(
            WebResult(
                title=data.get("Heading", "DuckDuckGo Answer"),
                url=data.get("AbstractURL", ""),
                snippet=data["AbstractText"],
                source="duckduckgo"
            )
        )
    return results

# ======================================
# WIKIPEDIA SEARCH
# ======================================
async def search_wikipedia(query: str):
    search_api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }

    async with httpx.AsyncClient(timeout=10) as client:
        data = (await client.get(search_api, params=params)).json()

    results = data.get("query", {}).get("search", [])
    if not results:
        return []

    first = results[0]
    title = first["title"]
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    summary_api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"

    async with httpx.AsyncClient(timeout=10) as client:
        summary = (await client.get(summary_api)).json()

    return [
        WebResult(
            title=title,
            url=url,
            snippet=summary.get("extract", ""),
            source="wikipedia"
        )
    ]

# ======================================
# MERGE SEARCH RESULTS
# ======================================
async def aggregate_search(query: str):
    google = await search_google(query)
    ddg = await search_duckduckgo(query)
    wiki = await search_wikipedia(query)

    all_results = google + ddg + wiki

    seen = set()
    final = []
    for r in all_results:
        if r.url not in seen:
            seen.add(r.url)
            final.append(r)

    return final[:7]

# ======================================
# BUILD FINAL ANSWER TEXT
# ======================================
def build_answer(q: str, results: List[WebResult]):
    if not results:
        return f"No results found for: {q}"

    text = f"Here's what I found about **{q}**:\n\n"
    for r in results[:3]:
        text += f"- {r.snippet} (source: {r.source})\n"
    return text

# ======================================
# MAIN AI ENDPOINT
# ======================================
@app.post("/api/crazeai", response_model=ChatResponse)
async def crazeai_chat(req: ChatRequest):
    q = req.message
    results = await aggregate_search(q)
    answer = build_answer(q, results)
    return ChatResponse(answer=answer, results=results)

# ======================================
# SERVE index.html
# ======================================
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# ======================================
# LOCAL RUN
# ======================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("crazeai_server:app", host="0.0.0.0", port=8000, reload=True)
