FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY . .
ENV PYTHONPATH=/app

RUN python -m app.agent_worker download-files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


CMD ["python", "-m", "app.agent_worker", "dev"]


# FROM python:3.11-slim

# WORKDIR /app



# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# COPY pyproject.toml uv.lock ./
# RUN uv pip install --system --no-cache -r pyproject.toml

# COPY . .

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

# EXPOSE 8001

# CMD ["python", "-m", "uvicorn", "app.fastapi_server:app", "--host", "0.0.0.0", "--port", "8001"]