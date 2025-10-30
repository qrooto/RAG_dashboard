import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('russian') | stopwords.words('english'))

def normalize_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def tokenize_for_bm25(t: str):
    t = t.lower()
    t = re.sub(r"[^a-zа-я0-9\s#@:_\-/\.]", " ", t)
    toks = [w for w in t.split() if w not in STOP and len(w) > 1]
    return toks
