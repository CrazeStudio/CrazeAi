from sentence_transformers import SentenceTransformer
import faiss

class WebAI:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.texts = []
        self.index = faiss.IndexFlatL2(384)

    def add_text(self, text):
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        embeddings = self.model.encode(chunks)

        self.index.add(embeddings)
        self.texts.extend(chunks)

    def ask(self, question):
        q_embedding = self.model.encode([question])
        _, result = self.index.search(q_embedding, 3)

        answers = [self.texts[i] for i in result[0]]
        return "\n".join(answers)
