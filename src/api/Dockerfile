# Stage 1
FROM python:3.10-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ga_core/ ./src/ga_core/

RUN pip install --no-cache-dir --user ".[api]"

# Stage 2
FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY src/api/ /app/src/api/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["fastapi", "run", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]