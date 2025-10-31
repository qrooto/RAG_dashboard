# Инструкция для запуска и работы

Эта заметка — краткий чек-лист: что сделать, куда положить файлы и как запустить.

---

## Вариант A: Локальный запуск (macOS / Linux, без Docker)

### 1) Установить Ollama локально
- **macOS**: скачать с [ollama.ai](https://ollama.ai) и установить (приложение запустится автоматически)
- Или через Homebrew: `brew install ollama`
- Подтянуть модель: `ollama pull llama3` (если ещё не скачана)
- Примечание: на macOS Ollama обычно работает как фоновый сервис, отдельно запускать `ollama serve` не нужно

### 2) Подготовить Python окружение
```bash
python3 -m venv .venv
source .venv/bin/activate  # на macOS/Linux
# или: .venv\Scripts\activate  # на Windows
pip install -r requirements.txt
```

### 3) Подготовить данные (CSV)
- Положите ваши CSV-файлы в папку `./data`.
- Обязательные столбцы: `chat_title, chat_username, message_id, message_link, date_utc, author_username, matched_keyword, text`

### 4) Запустить API локально
```bash
# Убедитесь что Ollama работает на http://127.0.0.1:11434
export OLLAMA_ENDPOINT=http://127.0.0.1:11434  # опционально, по умолчанию попробует локально
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 5) Загрузить CSV и задать вопрос
```bash
# Загрузить CSV в индекс
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"csv_paths": null, "overwrite": true}'

# Задать вопрос
python3 query_rag.py "кто пострадал из-за ftx и сколько потерял?"
```

---

## Вариант B: Docker Compose (Ubuntu 22, удалённый сервер)

### 1) Подготовка окружения (один раз)
- Требуется Ubuntu 22.04 с доступом к интернету.
- Установить Docker и Compose плагин:
```bash
bash scripts/setup_ubuntu.sh
# После установки Docker выйдите/войдите в систему (relogin),
# чтобы группа docker применилась для текущего пользователя
```

### 2) Подготовить данные (CSV)
- Положите ваши CSV-файлы в папку `./data` (рядом с этим файлом).
- Обязательные столбцы в CSV:
  - `chat_title, chat_username, message_id, message_link, date_utc, author_username, matched_keyword, text`
- Пример первой строки заголовков:
```csv
chat_title,chat_username,message_id,message_link,date_utc,author_username,matched_keyword,text
```
- Поле `text` — основной текст сообщения. Пустые значения по `text` игнорируются при загрузке.

### 3) Запустить весь стек (Ollama + API)
- Собрать и запустить контейнеры:
```bash
docker compose up -d --build
```
- Это поднимет два сервиса:
  - `ollama` — локальный сервер LLM, порт `11434`.
  - `api` — FastAPI сервис, порт `8000`.
- При первом старте автоматически подтянется модель `llama3` (через Ollama API).

### 4) Загрузить CSV в индекс (ингест)
- После старта сервисов выполните:
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"csv_paths": null, "overwrite": true}'
```
- Это возьмёт все `.csv` из `./data`, создаст эмбеддинги и сохранит индексы в `./index`.
- BM25-индекс будет пересобран автоматически.

### 5) Задать вопрос и получить ответ
- Пример запроса из CLI:
```bash
python3 query_rag.py "кто пострадал из-за ftx и сколько потерял?"
```
- Ответ придёт от RAG: API найдёт релевантные фрагменты из индекса и спросит модель `llama3`.

### 6) Полезные команды
- Логи API:
```bash
docker compose logs -f --tail=200 api
```
- Остановить стек:
```bash
docker compose down
```

---

## Общие заметки

### Частые вопросы / проблемы
- Долго качается модель: это нормально при первом старте; дождитесь загрузки `llama3`.
- Нет ответа от `/query`: убедитесь, что выполнили ингест, и сервисы работают (`docker compose ps` для Docker или проверьте процесс).
- CSV не находятся: проверьте, что файлы действительно лежат в `./data` и имеют расширение `.csv`.

### Кастомизация (необязательно)
- Модель LLM: по умолчанию `llama3`. Можно указать другую модель, если она доступна в Ollama, изменив модель в `app/rag.py` или через переменную `OLLAMA_ENDPOINT`.
- Повторный ре-ингест: в теле запроса `/ingest` используйте `{"overwrite": true}`, чтобы пересоздать коллекцию и пересобрать BM25.
