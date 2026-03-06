from fastapi import FastAPI, Query
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
async def embed(req: EmbedRequest | None = None, text: str | None = Query(None)):

    if req and req.text:
        input_text = req.text
    elif text:
        input_text = text
    else:
        return {"error": "No text provided"}

    input_text = input_text[:2000]

    embedding = model.encode(
        input_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).tolist()

    return {"embedding": embedding}