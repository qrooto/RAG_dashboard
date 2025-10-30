import sys, requests, json

def main():
    q = " ".join(sys.argv[1:]) or "кто писал о том, что потерял средства из-за FTX?"
    r = requests.post("http://127.0.0.1:8000/query", json={"query": q, "k": 5, "alpha": 0.6, "use_hybrid": True}, timeout=120)
    r.raise_for_status()
    js = r.json()
    print("=== ANSWER ===")
    print(js["answer"])
    print("\n=== HITS ===")
    for h in js["hits"]:
        print(f"{h['score']:.3f} | {h.get('date_utc')} | {h.get('chat_title')} | {h.get('author_username')} | {h.get('message_id')}")
        print(h["chunk"][:300], "\n")

if __name__ == "__main__":
    main()
