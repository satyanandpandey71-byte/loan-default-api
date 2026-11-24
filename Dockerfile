FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code + the trained model artifact (run `make train` before building).
COPY src/ ./src/
COPY models/ ./models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 1001 appuser && chown -R appuser /app
USER appuser

ENV MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=2).status == 200 else 1)"

CMD ["sh", "-c", "exec gunicorn --workers 2 --bind 0.0.0.0:${PORT} --access-logfile - src.wsgi:app"]
