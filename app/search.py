from typing import List, Dict, Tuple
import os, pickle
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from .utils import tokenize_for_bm25
from numpy import dot
from numpy.linalg import norm

CHROMA_PATH = "index/chroma"
COLL = "ftx_msgs"
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BM25_PATH = "index/bm25.pkl"

def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=False))
    return client.get_collection(COLL)

def build_bm25_index():
    coll = load_chroma()
    all_ids = coll.get(include=["metadatas"], where={})["ids"]
    docs = coll.get(ids=all_ids, include=["documents","metadatas"])
    corpus = [d for d in docs["documents"]]
    tokens = [tokenize_for_bm25(d) for d in corpus]
    bm25 = BM25Okapi(tokens)
    os.makedirs("index", exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": all_ids, "documents": corpus, "metas": docs["metadatas"]}, f)

def _load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

def cosine(a, b):
    return float(dot(a, b) / (norm(a) * norm(b) + 1e-12))

def semantic_search(query: str, k=5):
    coll = load_chroma()
    model = SentenceTransformer(EMB_NAME)
    qe = model.encode([query], normalize_embeddings=True)[0]
    res = coll.query(query_embeddings=[qe.tolist()], n_results=k, include=["documents","metadatas","distances"])
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "score": 1.0 - res["distances"][0][i],
            "chunk": res["documents"][0][i],
            "meta": res["metadatas"][0][i]
        })
    return hits

def hybrid_search(query: str, k=5, alpha=0.5, early_fusion=False):
    model = SentenceTransformer(EMB_NAME)
    qe = model.encode([query], normalize_embeddings=True)[0]

    bm = _load_bm25()
    tokens = tokenize_for_bm25(query)
    bm_scores = bm["bm25"].get_scores(tokens)
    topN = min(200, len(bm_scores))
    idxs = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:topN]

    if early_fusion:
        model = SentenceTransformer(EMB_NAME)
        import numpy as np
        cand_embs = model.encode([bm["documents"][i] for i in idxs], normalize_embeddings=True)
        sem_scores = [cosine(qe, e) for e in cand_embs]
        comb = [(i, alpha*sem_scores[j] + (1-alpha)*bm_scores[i]) for j,i in enumerate(idxs)]
        ranked = sorted(comb, key=lambda x: x[1], reverse=True)[:k]
        hits = []
        for i, sc in ranked:
            hits.append({
                "id": bm["ids"][i],
                "score": float(sc),
                "chunk": bm["documents"][i],
                "meta": bm["metas"][i]
            })
        return hits

    coll = load_chroma()
    sem_res = coll.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])
    sem_hits = {
        sem_res["ids"][0][i]: {
            "id": sem_res["ids"][0][i],
            "chunk": sem_res["documents"][0][i],
            "meta": sem_res["metadatas"][0][i],
            "sem": 1.0 - sem_res["distances"][0][i]
        } for i in range(len(sem_res["ids"][0]))
    }
    bm_hits = {}
    for rank, i in enumerate(idxs[:k*3]):
        bm_hits[bm["ids"][i]] = {
            "id": bm["ids"][i],
            "chunk": bm["documents"][i],
            "meta": bm["metas"][i],
            "bm25": bm_scores[i]
        }

    def _minmax(x, arr):
        lo, hi = float(min(arr)), float(max(arr))
        if hi - lo < 1e-9:
            return 0.0
        return (float(x) - lo) / (hi - lo)

    ids = set(sem_hits) | set(bm_hits)
    merged = []
    for _id in ids:
        s = sem_hits.get(_id, {})
        b = bm_hits.get(_id, {})
        sem = s.get("sem", 0.0)
        bmv = b.get("bm25", 0.0)
        sc = alpha*sem + (1-alpha)*_minmax(bmv, bm_scores)
        merged.append({
            "id": _id,
            "score": float(sc),
            "chunk": (s.get("chunk") or b.get("chunk")),
            "meta": (s.get("meta") or b.get("meta"))
        })
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:k]
