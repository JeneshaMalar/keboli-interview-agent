#!/bin/sh
echo "Downloading required files..."
python -m app.agent_worker download-files


echo "Starting Agent Worker..."
python -m app.agent_worker dev &

echo "Starting FastAPI Server..."
python -m uvicorn app.fastapi_server:app --host 0.0.0.0 --port 8001

wait