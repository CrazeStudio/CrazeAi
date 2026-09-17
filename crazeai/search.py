import html
import re
from urllib.parse import quote_plus
import httpx


def search_web(query: str, limit: int = 5):
    query = query.strip()
    if not query:
        return []
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    headers = {"User-Agent": "CrazeAI/1.0"}
    try:
        with httpx.Client(timeout=10, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception as exc:
        return [{"title": "Search unavailable", "url": "", "snippet": str(exc)}]
    pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.I | re.S)
    clean = lambda s: re.sub(r"\s+", " ", html.unescape(re.sub("<.*?>", "", s))).strip()
    return [{"title": clean(t), "url": html.unescape(u), "snippet": clean(s)} for u, t, s in pattern.findall(response.text)[:limit]]
