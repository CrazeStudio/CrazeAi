import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

# ======================================
# ENV VARIABLES FOR GOOGLE SEARCH
# ======================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# ======================================
# FASTAPI APP
# ======================================
app = FastAPI(title="CrazeAI")

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
    params = {"q": query, "key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "num": 3}

    async with httpx.AsyncClient(timeout=10.0) as client:
        data = (await client.get(url, params=params)).json()

    results = []
    for item in data.get("items", []):
        results.append(
            WebResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google",
            )
        )
    return results


# ======================================
# DUCKDUCKGO SEARCH
# ======================================
async def search_duckduckgo(query: str):
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}

    async with httpx.AsyncClient(timeout=10.0) as client:
        data = (await client.get(url, params=params)).json()

    results = []
    abstract = data.get("AbstractText")
    if abstract:
        results.append(
            WebResult(
                title=data.get("Heading", "DuckDuckGo Answer"),
                url=data.get("AbstractURL", ""),
                snippet=abstract,
                source="duckduckgo",
            )
        )
    return results


# ======================================
# WIKIPEDIA SEARCH
# ======================================
async def search_wikipedia(query: str):
    search_api = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        data = (await client.get(search_api, params=params)).json()

    results_list = data.get("query", {}).get("search", [])
    if not results_list:
        return []

    title = results_list[0]["title"]
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    summary_api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        summary = (await client.get(summary_api)).json()

    snippet = summary.get("extract", "")

    return [
        WebResult(
            title=title,
            url=page_url,
            snippet=snippet,
            source="wikipedia",
        )
    ]


# ======================================
# COMBINE SEARCH RESULTS
# ======================================
async def aggregate_search(query: str):
    google = await search_google(query)
    ddg = await search_duckduckgo(query)
    wiki = await search_wikipedia(query)

    all_res = google + ddg + wiki

    seen = set()
    unique = []
    for x in all_res:
        if x.url not in seen:
            seen.add(x.url)
            unique.append(x)

    return unique[:7]


# ======================================
# BUILD ANSWER
# ======================================
def build_answer(q, results):
    if not results:
        return f"No results found for: {q}"

    lines = [f"Here's what I found about **{q}**:\n"]
    for r in results[:3]:
        lines.append(f"- {r.snippet} (source: {r.source})")
    return "\n".join(lines)


# ======================================
# API ENDPOINT
# ======================================
@app.post("/api/crazeai", response_model=ChatResponse)
async def crazeai_chat(req: ChatRequest):
    q = req.message
    results = await aggregate_search(q)
    answer = build_answer(q, results)
    return ChatResponse(answer=answer, results=results)


# ======================================
# WEB UI (SERVES HTML)
# ======================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
  <head>
    <title>CrazeAI Web</title>
    <style>
      body { font-family: Arial; padding: 20px; background: #111; color: white; }
      input { width: 75%; padding: 10px; border-radius: 8px; border: none; }
      button { padding: 10px 20px; background: #00b7ff; border: none; border-radius: 8px; color: white; cursor: pointer; }
      .result-box { background: #222; padding: 10px; border-radius: 8px; margin-top: 10px; }
      a { color: #00b7ff; }
    </style>
  </head>
  <body>
    <h1>CrazeAI</h1>
    <input id="msg" placeholder="Ask me anything...">
    <button onclick="send()">Ask</button>

    <h2>Answer:</h2>
    <pre id="answer"></pre>

    <h2>Web Results:</h2>
    <div id="results"></div>

    <script>
      async function send() {
        const message = document.getElementById("msg").value;
        document.getElementById("answer").innerText = "Thinking...";
        document.getElementById("results").innerHTML = "";

        const res = await fetch("/api/crazeai", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message })
        });

        const data = await res.json();
        document.getElementById("answer").innerText = data.answer;

        data.results.forEach(r => {
          const box = document.createElement("div");
          box.className = "result-box";
          box.innerHTML = `
            <b>${r.title}</b><br>
            <p>${r.snippet}</p>
            <a href="${r.url}" target="_blank">Open</a> | Source: ${r.source}
          `;
          document.getElementById("results").appendChild(box);
        });
      }
    </script>
  </body>
</html>
"""


# ======================================
# LOCAL RUN
# ======================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("crazeai_server:app", host="0.0.0.0", port=8000, reload=True)
