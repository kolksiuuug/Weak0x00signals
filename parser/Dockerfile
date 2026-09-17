# Контекст сборки — КОРЕНЬ репозитория (context: .), чтобы в образ попал общий контракт /shared.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY parser/requirements.txt /app/parser/requirements.txt
RUN pip install --no-cache-dir -r /app/parser/requirements.txt

COPY shared/ /app/shared/
COPY parser/ /app/parser/

RUN useradd --create-home --uid 10003 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "parser.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
