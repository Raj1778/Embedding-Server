from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(text: str):
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    return {"embedding": embedding}