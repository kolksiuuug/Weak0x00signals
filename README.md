# Слабые сигналы

Сервис автоматического поиска «слабых сигналов» — ранних, малозаметных научно-технологических
трендов по открытым источникам. Хакатон Газпромбанк.Тех.

Правила проекта, контракт данных и границы ответственности участников — в [CLAUDE.md](CLAUDE.md).
**Читать обязательно перед первым коммитом.**

> **Статус: скелет-фундамент.** Все сервисы подняты и общаются end-to-end, но отдают
> **МОК-данные**. Реальные коннекторы к источникам, обученная модель и LLM подключаются
> участниками в своих каталогах.

---

## Архитектура

```
                 ┌────────────────────────────────────────────┐
  браузер ──80/443──▶ reverse-proxy (Caddy)                    │
                 │      /      → frontend:80  (nginx + React)  │
                 │      /api/* → api:8000     (FastAPI)        │
                 └────────────────┬───────────────────────────┘
                                  │ сеть edge
                          ┌───────▼────────┐
                          │  api (оркестр.) │  job_id + polling
                          └───┬────────┬────┘
                    сеть backend│        │
                ┌──────────────▼┐   ┌───▼───────────┐
                │ parser + LLM  │   │      ml       │
                └───────┬───────┘   └───────┬───────┘
                        │                   │
                  ┌─────▼─────┐       ┌─────▼─────┐
                  │  redis    │       │ postgres  │ (+ pgvector)
                  └───────────┘       └───────────┘
```

Наружу опубликован **только** `reverse-proxy` (порты 80/443). `api`, `ml`, `parser`,
`postgres`, `redis`, `frontend` портов на хост не отдают и доступны лишь внутри docker-сетей.

### Как сервисы получают общий контракт

`/shared/schema.py` — **граница слияния**: `SignalDoc`, `Source`, `SourceType`, `TrustLevel`
из §3 CLAUDE.md. Каждый python-сервис собирается с **контекстом = корень репозитория**
(`context: .` + `dockerfile: <сервис>/Dockerfile`) и копирует `shared/` внутрь образа:

```dockerfile
COPY shared/ /app/shared/
COPY api/    /app/api/
ENV PYTHONPATH=/app
```

Образ получается самодостаточным — его можно собрать в Portainer прямо из Git-репозитория,
bind-mount не требуется. Обратная сторона: **после правки `shared/` нужна пересборка**
(`docker compose build api ml parser`).

`/shared/contracts.py` — транспортные DTO (запросы/ответы HTTP). Их менять легче, но они
тоже общие для двух сервисов сразу.

---

## Запуск

Требуется Docker с плагином Compose v2.

```bash
git clone <url> && cd Weak0x00signals

cp .env.example .env          # Windows: copy .env.example .env
# обязательно задайте POSTGRES_PASSWORD в .env — без него compose не стартует

docker compose up -d --build
```

Откройте <http://localhost> → введите запрос → «Найти» → появится мок-выдача ТОП-15.

Swagger API: <http://localhost/api/docs>

### Полезные команды

```bash
docker compose ps                        # статус и healthcheck всех сервисов
docker compose logs -f api               # логи оркестратора
docker compose logs -f parser ml         # логи внутренних сервисов
docker compose build api ml parser       # пересборка после правки /shared
docker compose restart api               # перезапуск одного сервиса
docker compose down                      # остановить
docker compose down -v                   # остановить и снести тома (БД будет пересоздана)
```

### Проверка без браузера

> ⚠️ В Git Bash на Windows кириллица в аргументе `curl -d '...'` уходит в cp1251 и API
> отвечает `400 There was an error parsing the body`. Это ограничение консоли, не бага сервиса:
> используйте латиницу в запросе либо передавайте тело файлом `curl -d @body.json` в UTF-8.


```bash
curl http://localhost/api/health/deep                    # api + redis + ml + parser

JOB=$(curl -s -X POST http://localhost/api/search \
      -H 'Content-Type: application/json' \
      -d '{"query":"защита ИИ-моделей"}' | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

curl -s http://localhost/api/search/$JOB | python -m json.tool | head -60
curl -s http://localhost/api/signal/mock-003 | python -m json.tool
curl -s "http://localhost/api/sources?limit=5" | python -m json.tool
```

### Локальная разработка фронта без пересборки образа

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, /api проксируется на http://localhost (Caddy)
```

---

## Развёртывание на VPS (Portainer)

1. В `.env` на сервере указать домен без схемы: `DOMAIN=signals.example.ru`, `ACME_EMAIL=...`
   и реальный `POSTGRES_PASSWORD`.
2. Домен должен A-записью указывать на VPS — Caddy получит сертификат Let's Encrypt сам.
3. В Portainer: **Stacks → Add stack → Repository**, путь к `docker-compose.yml`,
   переменные окружения — через форму стека (не коммитить!).
4. На хосте должны быть свободны порты 80 и 443.

---

## API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | живость api |
| `GET` | `/api/health/deep` | живость api + redis + ml + parser |
| `POST` | `/api/search` | принять свободный запрос → `{job_id, poll_url}` (202) |
| `GET` | `/api/search/{job_id}` | статус и результат: ТОП-15 + отклонённые + статистика |
| `GET` | `/api/signal/{id}` | карточка-инсайт по одному сигналу |
| `GET` | `/api/sources` | все источники после дедупа и пересчёта доверенности |
| `GET` | `/api/docs` | Swagger UI |

Внутренние (только в docker-сети): `parser` — `POST /collect`, `GET /models`;
`ml` — `POST /score`, `POST /score/batch`.

---

## Структура репозитория

```
/
├── CLAUDE.md               # правила проекта и контракт — источник правды
├── docker-compose.yml
├── .env.example            # реальный .env НЕ коммитить
├── shared/                 # ГРАНИЦА СЛИЯНИЯ: общий контракт данных
│   ├── schema.py           #   §3 CLAUDE.md: SignalDoc, Source, enums
│   └── contracts.py        #   транспортные DTO между сервисами
├── api/                    # Участник 1: FastAPI-оркестратор, trust-слой, ИБ
├── ml/                     # Участник 2: модель, скоринг, фильтр зрелости, схема БД
├── parser/                 # Участник 3: коннекторы к источникам + LLM-сервис
├── frontend/               # Участник 1: React (Vite) + nginx
├── reverse-proxy/          # Caddyfile — единственный публичный вход
├── db/init.sql             # инициализация PostgreSQL + pgvector
├── data/                   # Участник 3: чистый датасет, кэш
└── docs/                   # схема архитектуры, отчёты по метрикам
```

## Границы работы

Каждый участник правит **только свой каталог**. `shared/`, `docker-compose.yml` и `CLAUDE.md` —
общие: изменения согласуются в чате команды.

| Каталог | Владелец |
|---|---|
| `api/`, `frontend/`, `reverse-proxy/`, корневые конфиги | Участник 1 |
| `ml/`, `db/` | Участник 2 |
| `parser/`, `data/` | Участник 3 |
| `shared/` | общий — только по согласованию |

## Что заглушено в скелете

| Место | Заглушка | Кто заменяет |
|---|---|---|
| `parser/app/mock_data.py` | 18 захардкоженных кандидатов с источниками | Участник 3 |
| `parser/app/llm.py` | `summarize_ru` / `translate_ru` / `generate_hypothesis` без сетевых вызовов | Участник 3 |
| `ml/app/main.py` | эвристика «тренд растёт + стадия ранняя + мало игроков/публикаций»; балл `стадия + тренд` выводится в `why`; фильтр зрелости по двум правилам | Участник 2 |
| `db/init.sql` | только `CREATE EXTENSION vector` | Участник 2 |
| `api/app/trust.py` | рабочий каркас trust-слоя, домены дополняются | Участник 1 |
