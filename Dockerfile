FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

FROM python:3.11-slim AS runtime

RUN groupadd -r appgroup && \
    useradd -r -g appgroup -m -u 1000 appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=appuser:appgroup . .

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

USER appuser

CMD ["/start.sh"]
