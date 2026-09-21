# Как запустить WEAK0x00 Signals

## Linux / macOS

1. Установите Docker с Compose v2.
2. В корне проекта выполните:

```bash
cp .env.example .env
nano .env
```

В `.env` обязательно задайте `POSTGRES_PASSWORD`, например:

```env
POSTGRES_PASSWORD=signals123
```

API-ключ LLM на первом запуске можно не указывать: проект использует grounded-шаблон для карточек, а live-поиск и скоринг работают отдельно. Для реального LLM-enrichment заполните одну из поддерживаемых конфигураций и оставьте `LLM_MODEL` из whitelist ТЗ.

3. Запустите весь стек:

```bash
docker compose up -d --build
```

4. Посмотрите состояние контейнеров:

```bash
docker compose ps
```

5. Проверьте backend:

```bash
curl http://localhost/api/health/deep
```

6. Откройте браузер:

```text
http://localhost
```

Swagger:

```text
http://localhost/api/docs
```

## Первый запуск ML

`ml-trainer` запускается один раз и читает `data/official_dataset.xlsx`. Сейчас официальный файл уже включён. Собственных негативов в `data/negative_candidates.csv` пока нет, поэтому trainer корректно завершится в режиме `heuristic`. Это не считается финальной ML-метрикой.

Чтобы получить LogisticRegression и воспроизводимые P/R/F1, заполните `data/negative_candidates.csv` собственными размеченными зрелыми/хайповыми/шумовыми кандидатами (минимум 10), затем выполните:

```bash
docker compose up -d --build ml-trainer
docker compose up -d ml
```

## Если что-то не запускается

Логи всех сервисов:

```bash
docker compose logs -f
```

Только API:

```bash
docker compose logs -f api
```

Parser:

```bash
docker compose logs -f parser
```

ML:

```bash
docker compose logs -f ml
```

Полностью пересоздать окружение (ВНИМАНИЕ: удалит БД и volumes):

```bash
docker compose down -v
docker compose up -d --build
```

## Windows PowerShell

Вместо `cp`:

```powershell
Copy-Item .env.example .env
```

Дальше команды `docker compose ...` такие же.
