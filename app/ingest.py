import os, uuid, pandas as pd
from typing import List, Dict
import logging, math
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from .utils import normalize_text
from datetime import datetime
import hashlib

CHROMA_PATH = "index/chroma"
COLL = "ftx_msgs"
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
logger = logging.getLogger("ingest")

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

    logger.info("Ingest started: overwrite=%s, csv_files=%d", overwrite, len(paths))
    model = SentenceTransformer(EMB_NAME)

    rows = []
    total_rows = 0
    for p in paths:
        df = pd.read_csv(p)
        logger.info("Reading CSV: %s (rows=%d)", p, len(df))
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
        total_rows += len(df)

    logger.info("Prepared chunks: total_rows=%d, raw_chunks=%d", total_rows, len(rows))

    seen = set()
    before = len(rows)
    rows = dedup_keep_first(rows, seen)
    logger.info("Deduplicated: kept=%d, removed=%d", len(rows), before - len(rows))

    B = 256
    total_batches = max(1, math.ceil(len(rows) / B))
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
        logger.info("Added batch %d/%d (size=%d)", (i // B) + 1, total_batches, len(batch))
    logger.info("Ingest finished: ingested=%d, csv_count=%d", len(rows), len(paths))
    return {"ingested": len(rows), "csv_count": len(paths)}
