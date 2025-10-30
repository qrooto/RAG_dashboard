#!/usr/bin/env bash
set -euo pipefail

# Wait for Ollama API
if [ -z "${OLLAMA_ENDPOINT:-}" ]; then
  export OLLAMA_ENDPOINT="http://ollama:11434"
fi

echo "Waiting for Ollama at ${OLLAMA_ENDPOINT}..."
for i in {1..60}; do
  if curl -s "${OLLAMA_ENDPOINT%/}/api/tags" > /dev/null; then
    break
  fi
  sleep 2
done

# Ensure llama3 model is present
echo "Ensuring llama3 model is available..."
curl -s -X POST "${OLLAMA_ENDPOINT%/}/api/pull" -d '{"name":"llama3"}' || true

echo "Starting API..."
exec uvicorn app.api:app --host 0.0.0.0 --port 8000


