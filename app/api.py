from fastapi import FastAPI
from .schema import QueryRequest, IngestRequest, RAGResponse, Hit
from .ingest import ingest_csvs
from .search import hybrid_search, semantic_search, build_bm25_index
from .rag import generate_answer

app = FastAPI(title="FTX RAG Dashboard (local)")

@app.post("/ingest")
def ingest(req: IngestRequest):
    out = ingest_csvs(req.csv_paths, overwrite=req.overwrite)
    build_bm25_index()
    return out

@app.post("/query", response_model=RAGResponse)
def query(req: QueryRequest):
    hits = hybrid_search(req.query, k=req.k, alpha=req.alpha) if req.use_hybrid else semantic_search(req.query, k=req.k)
    answer = generate_answer(req.query, hits)
    out_hits = [
        Hit(
            message_id=h["meta"].get("message_id",""),
            score=h["score"],
            chat_title=h["meta"].get("chat_title"),
            chat_username=h["meta"].get("chat_username"),
            author_username=h["meta"].get("author_username"),
            date_utc=h["meta"].get("date_utc"),
            matched_keyword=h["meta"].get("matched_keyword"),
            chunk=h["chunk"]
        )
        for h in hits
    ]
    meta = {"engine":"hybrid" if req.use_hybrid else "semantic", "alpha": str(req.alpha)}
    return RAGResponse(answer=answer, hits=out_hits, meta=meta)
