import os, pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLS_PATH = "index/victim_clf.pkl"

def train_classifier(labeled: List[Dict[str,str]]):
    model = SentenceTransformer(EMB_NAME)
    X = model.encode([r["text"] for r in labeled], normalize_embeddings=True)
    y = [r["label"] for r in labeled]
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    with open(CLS_PATH, "wb") as f:
        pickle.dump(clf, f)

def predict_batch(texts: List[str]):
    with open(CLS_PATH, "rb") as f:
        clf = pickle.load(f)
    model = SentenceTransformer(EMB_NAME)
    X = model.encode(texts, normalize_embeddings=True)
    return clf.predict_proba(X)[:,1]
