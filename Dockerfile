FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY README.md .

# Startup script pulls model (idempotent) and runs API
COPY scripts/start_api.sh /usr/local/bin/start_api.sh
RUN chmod +x /usr/local/bin/start_api.sh

ENV OLLAMA_ENDPOINT=http://ollama:11434

EXPOSE 8000

CMD ["/usr/local/bin/start_api.sh"]


