import torch


def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device=None):
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = x[:, -model.config.block_size:]
            logits, _ = model(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k and top_k < logits.size(-1):
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat((x, next_id), dim=1)
    return tokenizer.decode(x[0].tolist())
