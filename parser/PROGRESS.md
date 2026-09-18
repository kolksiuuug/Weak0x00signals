# PROGRESS.md — рабочий журнал ассистента (парсер, Участник 3 / Дима)

> Этот файл — память ассистента между сессиями. Обновляй его в конце каждой сессии:
> что сделано, что проверено, что дальше, и готовый промт для следующего запуска.
> Ветка: `dima` (CLAUDE.md §5 — не пушить в main напрямую, Conventional Commits).

---

## ПРОМТ НА СЛЕДУЮЩИЙ РАЗ (копируй целиком)

```
Прочитай PROGRESS.md из корня /parser и CLAUDE.md (§2, §3, §6 — я Дима/Data-Search Lead).
Ветка dima от main, работаю в /parser, чужие папки не трогаю, контракт POST /collect
(CollectResponse, shared/contracts.py) сохраняю стабильным.

Продолжи с того места, где остановился (см. PROGRESS.md → «Дальше по плану»).
Если после прочтения понял всё — просто начни с первого незавершённого пункта.
Соблюдай: Python 3.8+ (typing), все тексты/выдача на русском, LLM только из белого списка
(§2.4), каждый источник с полным набором метаданных (§2.7), моки в реальный /collect
не подмешивать (§2.3). В конце сессии обнови PROGRESS.md.
```

---

## Контекст проекта (коротко)

Сервис поиска «слабых сигналов» на хакатон Газпромбанк.Тех. Монорепо: `shared/` (граница
слияния — `schema.py`/`contracts.py`, менять только по согласованию команды), `api/` (У1),
`ml/` (У2), `parser/` (У3 — я), `data/`, `frontend/`, `reverse-proxy/`, `db/`.

Я — **Участник 3 (Дима, Data-Search Lead)**: коннекторы источников + LLM-сервис + датасет.
Мои каталоги: `parser/`, `data/`. Чужие (`api/`, `ml/`, `frontend/`, `shared/`, корневые
конфиги) **не трогаю**.

Контракт `POST /collect`: `CollectResponse(query, sources_processed, documents: List[SignalDoc])`
произвольное табло — **он уже реальный и менять структуру запрещено**.

## Текущий статус

- [x] 1. arXiv-коннектор + нормализация (модули готовы, проверены сквозно)
- [x] Redis-кэш (по запросу + по URL) с graceful degradation
- [ ] 2. OpenAlex / Crossref коннекторы
- [ ] 3. GDELT (новости) + PatentsView (патенты)
- [ ] 4. trafilatura (полный текст статей)
- [ ] 5. LLM-сервис: перевод запроса в англ. поисковые запросы, `summarize_ru`,
       `translate_ru` с пометкой «автоперевод» (§2.4/§2.9), `generate_hypothesis`
       строго по источникам (§2.3)
- [ ] 6. Финальный прогон через docker-compose
- [ ] 7. Обновить `parser/requirements.txt` если добавились зависимости (сейчас актуально)

---

## Сделано в этой сессии (файлы)

| Файл | Что делает |
|---|---|
| `parser/app/config.py` | Настройки из env: таймауты, ретраи, Redis URL+TTL, лимиты, URL API источников, User-Agent. Секретов нет (§5). |
| `parser/app/connectors/__init__.py` | `BaseConnector` (tenacity-ретраи только на 5xx/429/таймаут, `ConnectorError` для оркестратора), `registry`, `discover_connectors()` — ленивый импорт подключает модули коннекторов и регистрирует их. |
| `parser/app/connectors/arxiv.py` | arXiv Atom API (без ключа). Парсинг stdlib `xml.etree` (без feedparser). Каноничный URL без версии (`/abs/1303.3954`, а не `...v1`). Метаданные §2.7: `научная_статья`, `высокий`, `en`, ISO-дата. raw_text = аннотация. |
| `parser/app/normalize.py` | `canonical_url()` (http→https, без фрагмента, сортировка query), `dedupe_documents()` (по URL, слияние источников), `raw_to_signal_doc()`, `build_source()`, `merge_sources()`. Стабильный `id` = slug(title)-md5(url)[:8]. |
| `parser/app/cache.py` | Redis: `cache_get/set(+_json)`, `cache_or_compute`. Ключ = sha256. Пустой/упавший Redis → работаем без кэша, не падаем. |
| `parser/app/orchestrator.py` | `collect()`: фан-аут по всем коннекторам `asyncio.gather`, деградация per-connector, кэш по запросу и по URL, нормализация+дедуп, `CollectResponse` наружу. Моки не подмешиваются. |
| `parser/app/main.py` | `POST /collect` — реальный сбор; `GET /models`, `GET /health`, `GET /sources` (реальные источники посл. сбора). |
| `parser/tests/smoke_arxiv.py` | Проверка коннектора: `python -m parser.tests.smoke_arxiv "<запрос>"` |
| `parser/tests/smoke_collect.py` | Проверка оркестратора: `python -m parser.tests.smoke_collect "<запрос>"` |
| `parser/tests/smoke_api.py` | Проверка FastAPI через TestClient: /health /models /collect /sources |
| `parser/requirements.txt` | Добавлены `redis==5.2.1`, `trafilatura==1.10.0`. |

## Проверено в этой сессии

- `smoke_collect` и `smoke_api` на запросе `optical interconnect` → 10 документов, 200 OK.
- URL канонизируется (`1303.3954v1` → `/abs/1303.3954`), дедуп работает, метаданные полные.
- **Важно/известная особенность:** русский запрос в arXiv даёт **0 результатов** (arXiv
  англоязычный). Это решает LLM-шаг: перевод запроса в англ. поисковые слова до фан-аута.
- Локально на Windows только Python 3.14, поэтому venv `.venv` с новым pydantic (Docker
  соберётся под 3.11 с пинами из requirements — они ок).

## Паттерны, которые надо повторять в новых коннекторах

1. Класс наследует `BaseConnector`; задаёт `name`, `source_type`, `trust_level`,
   `language_default`, `timeout`.
2. Реализует `async def _search(self, query, limit) -> List[Dict[str, Any]]` — **без ретраев**,
   они приходят из базового `search()`.
3. Возвращает raw-dict: `doc_id, title, url, date(ISO|None), source_type, trust_level,
   language, raw_text, companies[], extra{}`.
4. Регистрирует себя: `registry.register(MyConnector())` на уровне модуля.
5. Оркестратор подхватит автоматически через `discover_connectors()` — **больше ничего
   менять не нужно**.

## Дальше по плану (порядок не менять!)

1. **OpenAlex** (`parser/app/connectors/openalex.py`): REST `api.openalex.org/works?search=...&per-page=50`,
   может `filter=from_publication_date:...`. Тип `научная_статья`, trust `высокий`, язык по
   `language` (если нет — `en`). Найти **Crossref**-данные если есть DOI. Автоконтекст: у
   OpenAlex есть `relevant_locations`, `authorships` (affiliations → можно выудить компании!).
2. **GDELT**: `https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=artlist&format=json`
   → `научно-популярная новость` → тип `новость`, trust `средний`. Язык из `language`.
3. **PatentsView**: `https://api.patentsview.org/patents/query?q={"_text_any":{"patent_abstract":"..."}}`.
   Тип `патент`, trust `высокий`. Патенты — сильный индикатор зарождающейся технологии.
4. **trafilatura** `parser/app/connectors/fetch_text.py`: для URL из найденных результатов
   докачивать полный текст → обогащать `raw_text`. `language` от `trafilatura` должен
   уехать в Source.
5. **LLM-сервис** (`parser/app/llm.py` — сейчас мок!): 
   - перевод запроса → англ. поисковые запросы (считай запрос с `LLM_MODEL`);
   - `summarize_ru(text, sources)` — резюме только по переданным источникам, иначе отказ;
   - `translate_ru(text, language)` — с префиксом `[автоперевод/генеративное резюме]`;
   - прокси к реальным API: GigaChat (scope/oauth), YandexGPT, OpenAI-совместимые;
   - **не забыть**: Docker env уже задаёт GIGACHAT_API_KEY/YANDEX_API_KEY/OPENAI_API_KEY.
6. Прогнать всё через `docker compose build parser && docker compose up -d` и smoke_api.

## Готовые команды для проверки

```bash
.venv\Scripts\python.exe parser\tests\smoke_collect.py "optical interconnect"
.venv\Scripts\python.exe parser\tests\smoke_api.py
```

## Правила-напоминание (не нарушать!)

- `shared/`, `CLAUDE.md`, `docker-compose.yml` — только через согласование команды.
- Моки в реальную выдачу не подмешивать (§2.3). `documents` из /collect = только реальный сбор.
- Каждый `Source` обязан иметь: title, url, date, source_type, language, trust_level (§2.7).
- Соцсети/блоги/пресс-релизы — пониженная доверенность, не единственное основание (§2.8).
- Зарубежные источники — русское резюме + пометка об автопереводе (§2.9).
- LLM только из белого списка (§2.4), выбор логируется.
- Ключи — только в env, никаких секретов в коде/коммитах (§5).
- Ветки `feat/...`, PR в main, Conventional Commits.