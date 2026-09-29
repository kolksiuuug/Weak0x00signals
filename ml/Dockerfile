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

# Обучение модели на сборке образа.
#
# Зачем. Артефакт ml/artifacts/model.joblib в git не хранится (.gitignore), поэтому в
# свежем клоне его нет. Без артефакта сервис поднимается, но работает на запасных
# правилах и честно сообщает об этом в /health (mock=true) — и recall@15 по открытому
# запросу падает с 90 % до 30 %. То есть стенд отвечает, отвечает 200, и отвечает плохо.
# Обучение детерминированное и занимает секунды, поэтому его место — сборка.
#
# До switch на appuser: артефакт пишется root-ом, следующий chown отдаёт его сервису.
#
# ВАЖНО: обучение требует датасет из data/raw, а корневой .dockerignore сейчас
# исключает data/raw из контекста сборки — значит этот шаг напечатает предупреждение
# и образ всё равно уедет в запасном режиме. Строку исключения нужно снять
# (зона Участника 1, согласуется в общем чате). Сборку намеренно не роняем: лучше
# работающий стенд с предупреждением в логе, чем упавшая сборка у коллеги.
COPY data/ /app/data/

ENV OFFICIAL_DATASET_PATH=/app/data/official_dataset.xlsx

RUN python -m ml.app.train

RUN useradd --create-home --uid 10002 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "ml.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
