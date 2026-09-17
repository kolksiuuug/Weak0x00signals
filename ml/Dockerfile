# Контекст сборки — КОРЕНЬ репозитория (context: .), чтобы в образ попал общий контракт /shared.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r /app/ml/requirements.txt

COPY shared/ /app/shared/
COPY ml/ /app/ml/

RUN useradd --create-home --uid 10002 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "ml.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
