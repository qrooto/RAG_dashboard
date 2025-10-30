# FTX RAG Dashboard (локальный)

Локальный стек: FastAPI + Chroma + Sentence-Transformers + Ollama (llama3).

## Быстрый старт
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Запуск API
uvicorn app.api:app --reload

# Загрузка CSV из ./data
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"csv_paths": null, "overwrite": true}'

# Пример запроса (CLI)
python query_rag.py "кто пострадал из-за ftx и сколько потерял?"
```

## Структура
```text
repo/
├── app/
│   ├── api.py
│   ├── search.py
│   ├── rag.py
│   ├── ingest.py
│   ├── classify.py
│   ├── schema.py
│   └── utils.py
├── data/            # положите сюда CSV
├── index/           # артефакты индекса (chroma, bm25, модели)
├── query_rag.py     # CLI-клиент к API
├── requirements.txt
└── README.md
```

## Формат CSV
Обязательные столбцы: `chat_title, chat_username, message_id, message_link, date_utc, author_username, matched_keyword, text`.

## Заметки
- При каждом `/ingest` пересобирается BM25-индекс.
- Дедупликация: SHA1 на уровне чанка. Для более агрессивного режима добавьте косинусное сравнение эмбеддингов.
- Классификатор "пострадал/нет" — baseline на логистической регрессии; обучите на своей разметке (см. `app/classify.py`).

