import os
import subprocess
from typing import List, Dict
import requests

def format_context(hits: List[Dict]) -> str:
    lines = []
    for h in hits:
        m = h["meta"] or {}
        head = f"[{m.get('chat_title','?')}] {m.get('author_username','?')} {m.get('date_utc','?')} (kw={m.get('matched_keyword','')})"
        lines.append(f"{head}\n{h['chunk']}\n")
    return "\n---\n".join(lines)

def ask_ollama_llama3(prompt: str) -> str:
    # По умолчанию пробуем локальный Ollama, в Docker будет переопределено через env
    endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    url = f"{endpoint.rstrip('/')}/api/generate"
    try:
        r = requests.post(url, json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=120)
        r.raise_for_status()
        js = r.json()
        return js.get("response", "")
    except Exception:
        # Fallback на subprocess (если HTTP API недоступен)
        proc = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode("utf-8"),
            capture_output=True
        )
        return proc.stdout.decode("utf-8", errors="ignore")

SYS = (
"You are an analyst. Answer strictly using the provided context. "
"If unknown, say you don't know. Return concise Russian output."
)

def generate_answer(query: str, hits: List[Dict]) -> str:
    ctx = format_context(hits)
    prompt = f"<<SYS>>\n{SYS}\n<</SYS>>\n\nВопрос: {query}\n\nКонтекст:\n{ctx}\n\nОтвет:"
    return ask_ollama_llama3(prompt)
