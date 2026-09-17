import re
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

FACTS = {
    "html": "HTML (HyperText Markup Language) is the standard markup language used to structure content on web pages. It defines elements such as headings, paragraphs, links, images, forms, and buttons.",
    "css": "CSS (Cascading Style Sheets) is used to control the appearance and layout of HTML pages, including colors, spacing, fonts, and responsive design.",
    "javascript": "JavaScript is a programming language commonly used to add interactive behavior and dynamic features to web pages and web applications.",
    "python": "Python is a general-purpose programming language known for readable syntax and a large ecosystem of libraries. It is widely used for automation, web development, data science, and AI.",
}

CODE = {
    "hello world": "print(\"Hello, World!\")",
    "hello world python": "print(\"Hello, World!\")",
}

def clean_key(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def direct_response(text):
    key = clean_key(text)
    if key in CASUAL: return CASUAL[key]
    if key in CODE: return "Python:\n```python\n" + CODE[key] + "\n```"
    if key in {"what is html", "what is html language", "explain html"}: return FACTS["html"]
    if key in {"what is css", "explain css"}: return FACTS["css"]
    if key in {"what is javascript", "what is js", "explain javascript"}: return FACTS["javascript"]
    if key in {"what is python", "explain python"}: return FACTS["python"]
    return None

def run(prompt, model, tokenizer, generate_fn, max_new_tokens=64):
    text = prompt.strip()
    direct = direct_response(text)
    if direct:
        return {"response": direct, "plan": {"intent":"direct","action":"direct","needs_search":False,"query":text}, "sources": []}

    plan = analyze(text)
    sources = []
    context = ""
    if plan["needs_search"]:
        sources = search_web(plan["query"], limit=3)
        usable = [s for s in sources if s.get("title") and s.get("snippet")]
        context = "\n\n".join(f"{s['title']}: {s['snippet'][:450]}" for s in usable)

    if context:
        model_prompt = f"Answer this question in 1-4 concise sentences using the web context. Never write a story. If the context is insufficient, say you don't have enough information.\nWeb context:\n{context}\n\nQuestion: {text}\nAnswer:"
    else:
        model_prompt = f"Give a short useful answer to the user's request. Never continue a fictional story.\nUser: {text}\nAnswer:"

    response = generate_fn(model, tokenizer, model_prompt, max_new_tokens=max_new_tokens)
    return {"response": response, "plan": plan, "sources": sources}
