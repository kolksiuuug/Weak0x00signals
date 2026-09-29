# WEAK0x00 Signals

Сервис для автоматизированного поиска и анализа зарождающихся научно-технологических трендов (слабых сигналов).

Финальный сценарий работает по **live открытым источникам**: стартовый датасет используется для обучения и оценки модели и не подмешивается в пользовательскую выдачу.

## Архитектура

```text
Пользователь
    │
    ▼
React → Caddy → FastAPI API
                  │
                  ├── Redis (jobs)
                  ├── Parser/RAG → arXiv / OpenAlex / Crossref
                  └── ML → Logistic Regression → maturity/hype → TOP-15
                                      │
                                      ▼
                               grounded enrichment
                                      │
                                      ▼
                              React report cards

PostgreSQL + pgvector хранит найденные документы,
источники и результаты.
```

## Локальный запуск

### Требования

* Docker Engine
* Docker Compose v2

### 1. Клонирование проекта

```bash
git clone <repository-url>
cd Weak0x00signals
```

### 2. Создание конфигурации

Скопируйте пример конфигурации:

```bash
cp .env.example .env
```

Откройте `.env`:

```bash
nano .env
```

Минимально необходимо задать:

```env
POSTGRES_PASSWORD=your-secure-password
POSTGRES_USER=signals
POSTGRES_DB=signals
```

Для использования LLM укажите ключ выбранного провайдера:

```env
LLM_MODEL=GigaChat 2 Max
GIGACHAT_API_KEY=
GIGACHAT_RQUID=
```

Если LLM не используется, соответствующие ключи можно оставить пустыми при условии, что конфигурация проекта допускает работу без них.

### 3. Запуск Docker Compose

Соберите и запустите все сервисы:

```bash
docker compose up -d --build
```

Проверьте состояние контейнеров:

```bash
docker compose ps
```

Все основные сервисы должны находиться в состоянии `Up`/`healthy`.

### 4. Проверка API

Проверьте состояние приложения:

```bash
curl http://localhost:18080/api/health/deep
```

При успешном запуске API должен вернуть информацию о состоянии основных компонентов системы.

### 5. Открытие интерфейса

Откройте в браузере:

```text
http://localhost:18080
```

Документация API Swagger:

```text
http://localhost:18080/api/docs
```

### 6. Остановка проекта

Для остановки контейнеров:

```bash
docker compose down
```

Для полной очистки контейнеров и данных проекта:

```bash
docker compose down -v
```

> `docker compose down -v` удаляет volumes проекта, включая данные PostgreSQL. Используйте эту команду только если данные больше не нужны.

## VPS с существующим Nginx Proxy Manager

Проект поддерживает безопасный режим развёртывания на VPS, где порты `80/443` уже используются существующим Nginx Proxy Manager.

В этом режиме проект не публикует свои сервисы напрямую на `80/443`. Caddy доступен локально через порт `18080`, а внешний доступ осуществляется через существующий Nginx Proxy Manager.

Создайте конфигурацию:

```bash
cp .env.example .env
nano .env
```

Укажите:

```env
POSTGRES_PASSWORD=your-secure-password
DOMAIN=your-domain.example
PROXY_NETWORK=weak-signals-proxy
NPM_CONTAINER=nginx_proxy_manager
LOCAL_PORT=18080
```

Запустите деплой:

```bash
./deploy-vps.sh
```

После запуска Nginx Proxy Manager должен проксировать указанный домен на:

```text
weak-signals:8080
```

по Docker-сети:

```text
weak-signals-proxy
```

Проект не требует освобождения портов `80/443` и не должен управлять или перезапускать другие Docker Compose-проекты на сервере.

Для локальной проверки самого reverse-proxy:

```bash
curl http://127.0.0.1:18080/api/health/deep
```

## Важно про пароль PostgreSQL

`POSTGRES_PASSWORD` должен быть задан в `.env`.

Файл `.env` содержит локальные секреты и **не должен добавляться в Git**.

Перед публикацией проекта убедитесь:

```bash
git check-ignore -v .env
```

## Обучение модели

Официальный датасет находится в:

```text
data/official_dataset.xlsx
```

Он используется **только на этапе обучения и оценки**.

Отрицательный класс зафиксирован отдельно в:

```text
data/control/mature_control.csv
```

В текущей сборке ML-модель обучается автоматически во время сборки Docker-образа сервиса `ml`. Отдельного `ml-trainer` сервиса нет.

После изменения датасета или кода обучения пересоберите ML-образ:

```bash
docker compose build --no-cache ml
docker compose up -d ml
```

Метрики находятся в:

```text
ml/reports/metrics.md
ml/reports/metrics.json
```

## Финальный open-search режим

Пользовательский поиск **не использует `official_dataset.xlsx` как fallback**.

Если live-коннекторы не нашли достаточно кандидатов, система не подменяет их обучающими примерами. Выдача формируется только из реально найденных открытых источников.

В разработке допускается:

```env
DEV_MOCK_FALLBACK=true
```

В `.env.example` и финальном стенде этот режим выключен:

```env
DEV_MOCK_FALLBACK=false
```

## Источники и доверенность

Для каждого найденного источника сохраняются:

* название;
* URL;
* дата, если доступна;
* тип;
* язык оригинала;
* уровень доверенности;
* отметка автоматического перевода или резюме.

Low-trust источники не могут быть единственным основанием для итогового кандидата.

## ML-интерпретируемость

Скоринг выполняет обученная Logistic Regression.

Для каждого результата API возвращает:

* `confidence`;
* `why`;
* ключевые предикторы;
* версию модели;
* режим модели.

Фильтр зрелости/hype формирует `rejected_reason` для исключённых кандидатов.

## API

| Метод | Путь                   | Назначение                               |
| ----- | ---------------------- | ---------------------------------------- |
| GET   | `/api/health/deep`     | Проверка API, Redis, parser и ML         |
| POST  | `/api/search`          | Запуск свободного live-поиска            |
| GET   | `/api/search/{job_id}` | Получение результата TOP-15              |
| GET   | `/api/signal/{id}`     | Получение отдельного report-like инсайта |
| GET   | `/api/sources`         | Получение сохранённых источников         |

## LLM

Используется одна явно выбранная модель из whitelist ТЗ через `LLM_MODEL`.

Автоматического выбора модели и OpenRouter нет.

LLM получает только найденный контекст и источники. Она не определяет `confidence` и не является единственным основанием для включения сигнала.

Без API-ключа система сохраняет live-поиск и ML-скоринг и может использовать grounded-template для текстовой карточки, если это разрешено текущей конфигурацией.

Для финальной демонстрации необходимо указать ключ выбранного разрешённого провайдера.

## Проверки

Проверка Python-модулей:

```bash
python -m compileall shared api ml parser
```

Проверка структуры проекта:

```bash
python docs/check_project.py
```

Сборка frontend:

```bash
cd frontend
npm ci
npm run build
cd ..
```

## Docker smoke test

Для полной проверки Docker-окружения:

```bash
docker compose down -v
docker compose up -d --build
docker compose ps
curl http://localhost:18080/api/health/deep
```

После запуска откройте интерфейс:

```text
http://localhost:18080
```

и выполните несколько различных открытых поисковых запросов.

В результатах поиска необходимо убедиться, что используется:

```text
retrieval_mode=live
```

а результаты содержат реальные источники.
