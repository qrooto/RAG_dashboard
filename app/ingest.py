import os, uuid, pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from .utils import normalize_text
from datetime import datetime
import hashlib

CHROMA_PATH = "index/chroma"
COLL = "ftx_msgs"
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text: str, size=500, overlap=50):
    text = normalize_text(text)
    res = []
    i = 0
    while i < len(text):
        res.append(text[i:i+size])
        i += size - overlap
    return res or [""]

def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def dedup_keep_first(rows: List[Dict], seen: set) -> List[Dict]:
    out = []
    for r in rows:
        h = text_hash(r["chunk"])
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=False))
    if COLL not in [c.name for c in client.list_collections()]:
        return client.create_collection(COLL, metadata={"hnsw:space":"cosine"})
    return client.get_collection(COLL)

def ingest_csvs(paths: List[str] | None = None, overwrite: bool=False):
    os.makedirs("index", exist_ok=True)
    coll = get_collection()
    if overwrite:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(COLL)
        coll = client.create_collection(COLL, metadata={"hnsw:space":"cosine"})

    if not paths:
        paths = [os.path.join("data", p) for p in os.listdir("data") if p.endswith(".csv")]
    assert paths, "Нет CSV для загрузки"

    model = SentenceTransformer(EMB_NAME)

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            text = str(r.get("text", "") or "")
            if not text.strip():
                continue
            chunks = chunk_text(text)
            for ch in chunks:
                rows.append({
                    "id": f"{r.get('message_id')}-{uuid.uuid4().hex[:8]}",
                    "chunk": ch,
                    "meta": {
                        "chat_title": r.get("chat_title"),
                        "chat_username": r.get("chat_username"),
                        "message_id": r.get("message_id"),
                        "message_link": r.get("message_link"),
                        "date_utc": r.get("date_utc"),
                        "author_username": r.get("author_username"),
                        "matched_keyword": r.get("matched_keyword"),
                        "source_csv": os.path.basename(p),
                        "ingested_at": datetime.utcnow().isoformat()
                    }
                })

    seen = set()
    rows = dedup_keep_first(rows, seen)

    B = 256
    for i in range(0, len(rows), B):
        batch = rows[i:i+B]
        texts = [r["chunk"] for r in batch]
        embs = model.encode(texts, normalize_embeddings=True)
        coll.add(
            ids=[r["id"] for r in batch],
            documents=texts,
            embeddings=embs.tolist(),
            metadatas=[r["meta"] for r in batch]
        )
    return {"ingested": len(rows), "csv_count": len(paths)}
