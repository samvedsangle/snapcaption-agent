FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api.py .
COPY snapcaption ./snapcaption

# Run as an unprivileged user; /data holds the SQLite file.
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
USER app

# Mount a volume at /data if you want results to survive redeploys.
ENV PORT=7860 DB_PATH=/data/snapcaption.db
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
