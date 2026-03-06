from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import os


torch.set_grad_enabled(False)


torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI()

model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2",
    device="cpu"
)

class EmbedRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(req: EmbedRequest):
    embedding = model.encode(
        req.text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).tolist()

    return {"embedding": embedding}