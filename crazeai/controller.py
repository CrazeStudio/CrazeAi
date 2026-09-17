from .analyzer import analyze
from .search import search_web

CASUAL = {
    "hi": "Hi! I'm CrazeAI. How can I help?",
    "hello": "Hello! I'm CrazeAI. What can I help you with?",
    "hey": "Hey! What can I help you with?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "good morning": "Good morning! How can I help?",
    "good night": "Good night!",
}

def run(prompt, model, tokenizer, generate_fn, max_new_tokens=64):
    text = prompt.strip()
    casual = CASUAL.get(text.lower())
    if casual:
        return {"response": casual, "plan": {"intent": "casual_chat", "action": "direct", "needs_search": False, "query": text}, "sources": []}

    plan = analyze(text)
    sources = []
    context = ""
    if plan["needs_search"]:
        sources = search_web(plan["query"], limit=3)
        usable = [s for s in sources if s.get("title") and s.get("snippet")]
        context = "\n\n".join(f"{s['title']}: {s['snippet'][:500]}" for s in usable)

    if context:
        model_prompt = f"Answer the user concisely using only this web context.\n{context}\n\nUser: {text}\nAssistant:"
    else:
        model_prompt = f"Answer the user naturally and concisely. Do not continue a story.\nUser: {text}\nAssistant:"

    response = generate_fn(model, tokenizer, model_prompt, max_new_tokens=max_new_tokens)
    return {"response": response, "plan": plan, "sources": sources}
