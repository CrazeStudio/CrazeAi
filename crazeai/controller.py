from .analyzer import analyze
from .search import search_web


def run(prompt, model, tokenizer, generate_fn, max_new_tokens=96):
    plan = analyze(prompt)
    sources = []
    context = ""

    if plan["needs_search"]:
        sources = search_web(plan["query"], limit=4)
        usable = [s for s in sources if s.get("title") and s.get("snippet")]
        context = "\n\n".join(
            f"Source: {s['title']}\n{s['snippet']}" for s in usable
        )

    if context:
        model_prompt = (
            "Answer the user using the web context below. Be concise. "
            "If the context does not support an answer, say so.\n\n"
            f"Web context:\n{context}\n\nUser: {prompt}\nAssistant:"
        )
    else:
        model_prompt = f"User: {prompt}\nAssistant:"

    text = generate_fn(model, tokenizer, model_prompt, max_new_tokens=max_new_tokens)
    return {"response": text, "plan": plan, "sources": sources}
