"""
Chroma-backed vector store for the saree catalogue, with a colour-aware
re-ranking stage layered on top of plain ANN search.

SEARCH QUALITY STRATEGY (two-stage retrieve-then-rerank)
----------------------------------------------------------
1. ANN retrieve `candidate_k` nearest neighbours by DINOv2 cosine similarity.
   This is fast and gets us a shortlist that's already texture/pattern-aware.
2. Re-rank that shortlist by a weighted blend of DINOv2 similarity and HSV
   colour-histogram similarity, then keep the top_k.

Doing the blend as a *re-rank* rather than concatenating both vectors into
one indexed embedding means we can freely tune dino_weight without ever
having to re-embed or re-index the catalogue.
"""
import json
import os
import chromadb
import numpy as np
from PIL import Image

from core.embeddings import get_dino_embedding, get_color_histogram, histogram_similarity

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = "sarees"


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_image(collection, image_id: str, image_path: str, image_url: str = None):
    img = Image.open(image_path)
    dino_vec = get_dino_embedding(img)
    color_hist = get_color_histogram(img)
    collection.add(
        ids=[image_id],
        embeddings=[dino_vec.tolist()],
        metadatas=[{
            "path": image_path,
            "url": image_url or "",
            "color_hist": json.dumps(color_hist.tolist()),
        }],
    )


def search(collection, query_img: Image.Image, top_k: int = 5,
           candidate_k: int = 30, dino_weight: float = 0.65):
    query_vec = get_dino_embedding(query_img)
    query_hist = get_color_histogram(query_img)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(candidate_k, max(collection.count(), 1)),
        include=["metadatas", "distances"],
    )

    ids = results["ids"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance = 1 - cosine_sim

    scored = []
    for _id, meta, dist in zip(ids, metas, distances):
        dino_sim = 1 - dist
        cand_hist = np.array(json.loads(meta["color_hist"]), dtype=np.float32)
        color_sim = histogram_similarity(query_hist, cand_hist)
        final_score = dino_weight * dino_sim + (1 - dino_weight) * color_sim
        scored.append({
            "id": _id,
            "path": meta["path"],
            "url": meta.get("url", ""),
            "score": round(float(final_score), 4),
            "dino_similarity": round(float(dino_sim), 4),
            "color_similarity": round(float(color_sim), 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
