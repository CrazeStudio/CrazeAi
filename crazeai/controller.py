from .analyzer import analyze
from .search import search_web


def run(prompt, model, tokenizer, generate_fn, max_new_tokens=120):
    plan = analyze(prompt)
    sources = []
    context = ""
    if plan["needs_search"]:
        sources = search_web(plan["query"], limit=5)
        usable = [s for s in sources if s.get("title") and s.get("snippet")]
        context = "\n\n".join(f"Source: {s['title']}\n{s['snippet']}\nURL: {s['url']}" for s in usable)
    model_prompt = ("Use the following search results as context. Answer the user briefly and do not invent unsupported facts.\n\n" + context + f"\n\nUser: {prompt}\nAssistant:") if context else f"User: {prompt}\nAssistant:"
    text = generate_fn(model, tokenizer, model_prompt, max_new_tokens=max_new_tokens)
    return {"response": text, "plan": plan, "sources": sources}
