from .analyzer import analyze
from .search import search_web


def run(prompt, model, tokenizer, generate_fn, max_new_tokens=64):
    plan = analyze(prompt)
    sources = []
    context = ""
    if plan["needs_search"]:
        sources = search_web(plan["query"], limit=3)
        usable = [s for s in sources if s.get("title") and s.get("snippet")]
        context = "\n\n".join(f"Source: {s['title']}\n{s['snippet'][:500]}" for s in usable)
    if context:
        model_prompt = f"Use this web context. Answer concisely and don't invent facts.\n\n{context}\n\nUser: {prompt}\nAssistant:"
    else:
        model_prompt = f"User: {prompt}\nAssistant:"
    text = generate_fn(model, tokenizer, model_prompt, max_new_tokens=max_new_tokens)
    return {"response": text, "plan": plan, "sources": sources}
