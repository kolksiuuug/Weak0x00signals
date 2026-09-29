# Запуск WEAK0x00 Signals

## Linux / macOS

1. Установите Docker Engine с Compose v2.
2. В корне проекта: 

```bash
cp .env.example .env
nano .env
```

Обязательно оставьте непустой `POSTGRES_PASSWORD`. API-ключ LLM можно добавить позже; без него live-поиск и ML работают, а текстовая часть использует grounded-template.

3. Соберите и запустите весь стек:

```bash
docker compose up -d --build
```

ML-модель обучается во время сборки `ml`-образа. **Отдельного `ml-trainer` сервиса нет.**

4. Проверка:

```bash
docker compose ps
curl http://localhost/api/health/deep
```

5. Откройте:

```text
http://localhost
```

Swagger:

```text
http://localhost/api/docs
```

## Важный финальный режим

Пользовательский open search использует только live открытые источники. `data/official_dataset.xlsx` и `data/control/mature_control.csv` нужны для обучения/оценки ML и **не подмешиваются в live-выдачу**.

Если live-источники временно не дали достаточно кандидатов, система не заполняет выдачу обучающими примерами. Это намеренно соответствует финальному ТЗ.

## Пересборка модели

После изменения `data/official_dataset.xlsx`, `data/control/` или кода ML:

```bash
docker compose build --no-cache ml
docker compose up -d ml
```

## Логи

```bash
docker compose logs -f api
docker compose logs -f parser
docker compose logs -f ml
```

## Полный сброс

ВНИМАНИЕ: удаляет PostgreSQL/Redis volumes.

```bash
docker compose down -v
docker compose up -d --build
```

## Windows PowerShell

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

## LLM для финальной демонстрации

Выберите одну разрешённую модель через `LLM_MODEL` и задайте её ключи в `.env`. Автоматического выбора модели нет. Не добавляйте ключи в Git.

После изменения LLM-конфигурации:

```bash
docker compose up -d --build parser
```

В логах parser должен быть виден выбранный провайдер/модель и результат grounded enrichment.

## VPS с существующим Nginx Proxy Manager

Если 80/443 уже заняты NPM, не запускайте старый режим с публикацией Caddy на 80/443. Используйте:

```bash
cp .env.example .env
# задайте POSTGRES_PASSWORD и DOMAIN
./deploy-vps.sh
```

NPM должен проксировать `your-domain.example` на `weak-signals:8080` по сети `weak-signals-proxy`. 
