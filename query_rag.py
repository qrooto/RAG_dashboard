import sys, requests, json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, nargs="+", help="Текст запроса")
    parser.add_argument("--k", type=int, default=5, help="Сколько документов вернуть (<=0 — все)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Смесь семантики/BM25 (0..1)")
    parser.add_argument("--no-hybrid", action="store_true", help="Отключить гибридный поиск (только семантика)")
    args = parser.parse_args()

    q = " ".join(args.query)
    payload = {"query": q, "k": args.k, "alpha": args.alpha, "use_hybrid": (not args.no_hybrid)}
    r = requests.post("http://127.0.0.1:8000/query", json=payload, timeout=600)
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
