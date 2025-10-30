from pydantic import BaseModel
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    alpha: float = 0.5
    use_hybrid: bool = True

class IngestRequest(BaseModel):
    csv_paths: Optional[List[str]] = None
    overwrite: bool = False

class Hit(BaseModel):
    message_id: str
    score: float
    chat_title: Optional[str] = None
    chat_username: Optional[str] = None
    author_username: Optional[str] = None
    date_utc: Optional[str] = None
    matched_keyword: Optional[str] = None
    chunk: str

class RAGResponse(BaseModel):
    answer: str
    hits: List[Hit]
    meta: Dict[str, str] = {}
