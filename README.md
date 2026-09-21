# WEAK0x00 Signals

Сервис автоматического поиска ранних научно-технологических трендов по открытым источникам.

Проект реализует фактическую архитектуру ТЗ: **свободный запрос → live-поиск → нормализация источников → trust-слой → интерпретируемый скоринг → фильтр зрелости/хайпа → RAG-enrichment → ТОП-15**.

## Данные

Официальный XLSX на 100 сигналов уже лежит в `data/official_dataset.xlsx`.

Важно: в датасете заказчика нет негативов. Поэтому без заполненного `data/negative_candidates.csv` сервис запускается с прозрачной fallback-эвристикой. Для финальных P/R/F1 нужно добавить собственные негативные кандидаты команды; это соответствует актуальному Q&A организаторов.

Аудит несоответствий и список оставшихся действий: [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).

## Архитектура

```text
Browser
  │ HTTPS
  ▼
Caddy ── /api/* ──► FastAPI API ──► Redis jobs
  │                       │
  │                       ├──► Parser/RAG ──► arXiv/OpenAlex/Crossref/GDELT/PatentsView
  │                       │                   └── PostgreSQL + pgvector
  │                       └──► ML ──► LogisticRegression / heuristic fallback
  │                                    ↑
  └──────────────► React + nginx       └── trained model volume
```

## Запуск

```bash
cp .env.example .env
# ОБЯЗАТЕЛЬНО задать POSTGRES_PASSWORD

docker compose up -d --build
```

Локально: `http://localhost`.
Swagger: `http://localhost/api/docs`.

Для VPS укажите `DOMAIN=signals.example.ru` и `ACME_EMAIL=...`. Caddy сам выпускает TLS-сертификат.

## Переменные LLM

`LLM_MODEL` должен быть одним из разрешённых ТЗ. Автовыбор моделей не используется.

- GigaChat: `GIGACHAT_API_KEY`, при необходимости `GIGACHAT_RQUID`
- YandexGPT: `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`
- gpt-4.1 / gpt-5.6-luna / Qwen: `OPENAI_API_KEY` и при необходимости свой `OPENAI_BASE_URL`

Секреты не коммитить.

## Данные для обучения

Официальный файл уже включён в репозиторий:

```text
data/official_dataset.xlsx
```

Лист: `Слабые сигналы`. В файле первая строка — презентационный заголовок, поэтому загрузчик читает названия колонок со второй строки.

Негативы:

```text
data/negative_candidates.csv
```

Запуск обучения происходит отдельным compose-сервисом `ml-trainer` и сохраняет модель в volume `ml_model`.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | API health |
| GET | `/api/health/deep` | API + Redis + parser + ML |
| POST | `/api/search` | свободный запрос |
| GET | `/api/search/{job_id}` | статус/результат ТОП-15 |
| GET | `/api/signal/{id}` | реальная карточка сигнала |
| GET | `/api/sources` | сохранённые источники |

Внутри Docker:

- parser: `/collect`, `/enrich`, `/document/{id}`, `/sources`, `/models`
- ml: `/score`, `/score/batch`, `/metrics`

## Проверка

Без Docker можно проверить синтаксис:

```bash
python -m compileall shared api ml parser
```

Функциональные тесты live-коннекторов требуют доступа к внешней сети.
