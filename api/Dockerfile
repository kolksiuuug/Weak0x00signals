# Контекст сборки — КОРЕНЬ репозитория (см. docker-compose.yml: context: .),
# чтобы в образ попал общий контракт /shared. Образ самодостаточен: bind-mount не нужен.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY shared/ /app/shared/
COPY api/ /app/api/

# Сервис не работает от root (ИБ)
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
