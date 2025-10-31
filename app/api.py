from fastapi import FastAPI
import logging
from .schema import QueryRequest, IngestRequest, RAGResponse, Hit
from .ingest import ingest_csvs
from .search import hybrid_search, semantic_search, build_bm25_index
from .rag import generate_answer

app = FastAPI(title="FTX RAG Dashboard (local)")
log = logging.getLogger("api")

@app.get("/")
def root():
    return {"status": "ok", "message": "FTX RAG API. Use /ingest and /query"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    out = ingest_csvs(req.csv_paths, overwrite=req.overwrite)
    build_bm25_index()
    return out

@app.post("/query", response_model=RAGResponse)
def query(req: QueryRequest):
    try:
        if req.use_hybrid:
            try:
                hits = hybrid_search(req.query, k=req.k, alpha=req.alpha)
                engine = "hybrid"
            except Exception as e:
                log.warning("Hybrid search failed, fallback to semantic: %s", e)
                hits = semantic_search(req.query, k=req.k)
                engine = "semantic"
        else:
            hits = semantic_search(req.query, k=req.k)
            engine = "semantic"
    except Exception as e:
        log.exception("Search failed: %s", e)
        hits = []
        engine = "error"

    try:
        answer = generate_answer(req.query, hits)
    except Exception as e:
        log.exception("LLM generation failed: %s", e)
        answer = "Не удалось получить ответ от LLM. Проверьте, что Ollama запущен (и модель llama3 доступна). Результаты поиска приложены."
    out_hits = [
        Hit(
            message_id=str(h["meta"].get("message_id", "")),
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
    meta = {"engine": engine, "alpha": str(req.alpha)}
    return RAGResponse(answer=answer, hits=out_hits, meta=meta)
