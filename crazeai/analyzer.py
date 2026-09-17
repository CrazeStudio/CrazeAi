import re

SEARCH_WORDS = {"latest", "today", "current", "news", "recent", "weather", "price", "score", "update", "updates", "search"}


def analyze(prompt: str):
    text = prompt.strip()
    lower = text.lower()
    needs_search = any(word in lower for word in SEARCH_WORDS)
    if re.search(r"\b(when|where|who|what|which)\b", lower) and any(x in lower for x in ("2026", "today", "latest", "current", "recent")):
        needs_search = True
    intent = "question" if text.endswith("?") or lower.startswith(("what ", "why ", "how ", "who ", "where ", "when ")) else "conversation"
    return {"intent": intent, "needs_search": needs_search, "action": "search_and_answer" if needs_search else "answer", "query": text}
