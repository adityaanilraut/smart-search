FROM ghcr.io/astral-sh/uv:0.9.28 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HOME=/models
CMD ["uvicorn", "smart_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
