import re

SEARCH_PATTERNS = (
    r"\b(latest|today|current|recent|news|weather|price|prices|score|scores|update|updates|release|version|stock|market)\b",
    r"\b(what is|who is|where is|when is|how much|how many)\b.*\b(today|now|current|latest|recent)\b",
)
QUESTION_STARTS = ("what ", "why ", "how ", "who ", "where ", "when ", "which ", "can ", "is ", "are ", "do ", "does ", "will ")


def analyze(prompt: str):
    text = re.sub(r"\s+", " ", prompt.strip())
    lower = text.lower()
    is_question = text.endswith("?") or lower.startswith(QUESTION_STARTS)
    needs_search = any(re.search(pattern, lower) for pattern in SEARCH_PATTERNS)

    # Explicit web requests always use the web route.
    if re.search(r"\b(search|look up|find online|browse the web|on the internet)\b", lower):
        needs_search = True

    if needs_search:
        action = "search_and_answer"
    elif is_question:
        action = "answer"
    else:
        action = "chat"

    return {
        "intent": "question" if is_question else "conversation",
        "needs_search": needs_search,
        "action": action,
        "query": text,
    }
