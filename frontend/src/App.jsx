import { useMemo, useState } from 'react'
import { startSearch, pollUntilDone, fetchSignal } from './api.js'

const STAGE_LABELS = { 1: 'Исследование', 2: 'Прототип / PoC', 3: 'Пилот', 4: 'Раннее внедрение' }
const TREND_LABELS = { 1: 'стабильно', 2: 'растёт', 3: 'растёт быстро' }
const EXAMPLES = ['защита ИИ-моделей', 'роботы для промышленности', 'инфраструктура для ИИ']

const STOP_WORDS = new Set(['для', 'и', 'в', 'на', 'по', 'из', 'с', 'до', 'как', 'the', 'of', 'for', 'and', 'in', 'to', 'with'])
const TOPIC_ALIASES = [
  ['ии', 'ai', 'artificial', 'intelligence'],
  ['промышлен', 'industrial', 'manufactur', 'factory', 'production', 'завод', 'производств'],
  ['робот', 'robot', 'automation', 'автоматизац'],
  ['кибер', 'cyber', 'security', 'безопасн'],
  ['квант', 'quantum'],
  ['данн', 'data', 'dataset', 'аналит'],
  ['энерг', 'energy', 'power', 'электро'],
  ['медицин', 'health', 'medical', 'клинич'],
]

function tokenize(value = '') {
  return String(value).toLowerCase().replace(/ё/g, 'е').match(/[a-zа-я0-9]{3,}/g)?.filter((token) => !STOP_WORDS.has(token)) || []
}

function topicMatches(token, alias) {
  return alias.some((part) => token.includes(part) || part.includes(token))
}

// UI-защита от очевидно нерелевантных карточек при деградации LLM/rerank.
// Она не пересчитывает ML-score и не подменяет серверный ranking: только убирает
// документы, в которых нет ни одного признака темы запроса.
function isRelevantToQuery(doc, query) {
  const queryTokens = tokenize(query)
  if (queryTokens.length === 0) return true
  const haystack = tokenize([
    doc.title,
    ...(doc.companies || []),
    doc.description,
    doc.raw_text,
    doc.evidence_summary,
  ].filter(Boolean).join(' '))

  const groups = TOPIC_ALIASES.filter((group) =>
    queryTokens.some((q) => group.some((part) => q.includes(part) || part.includes(q)))
  )
  if (groups.length > 1) {
    return groups.every((group) => haystack.some((h) => group.some((part) => h.includes(part) || part.includes(h))))
  }
  return queryTokens.some((q) => haystack.some((h) => h.includes(q) || q.includes(h)))
}

function stableScore(value) {
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : null
}

function Icon({ name, size = 18 }) {
  const paths = {
    search: <><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4 4" /></>,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" /></>,
    external: <><path d="M14 5h5v5" /><path d="m19 5-8 8" /><path d="M19 13v5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5" /></>,
    database: <><ellipse cx="12" cy="5" rx="7" ry="3" /><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5" /><path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="8" /><path d="M12 8v5l3 2" /></>,
    orbit: <><circle cx="12" cy="12" r="7" /><circle cx="12" cy="12" r="2.2" /><path d="M5 6.8C7.2 4.4 10.2 3 13.2 3c4.1 0 7.5 2.7 7.8 6.4" /></>,
    filter: <><path d="M4 6h16" /><path d="M7 12h10" /><path d="M10 18h4" /></>,
    close: <><path d="m6 6 12 12" /><path d="M18 6 6 18" /></>,
  }
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}

const MODEL_MODE_LABELS = {
  trained: 'обученный',
  rule_fallback: 'резервный правиловый',
}

function modelModeLabel(value) {
  return MODEL_MODE_LABELS[value] || value || 'обученный'
}

function Score({ value, compact = false }) {
  const normalized = stableScore(value)
  if (normalized == null) return <span className="score muted-score">—</span>
  const percent = normalized * 100
  return (
    <div className={`score-wrap ${compact ? 'score-compact' : ''}`}>
      <div className="score-number">{percent.toFixed(1)}<small>%</small></div>
      <div className="score-bar"><i style={{ width: `${percent}%` }} /></div>
      <span className="score-caption">уверенность модели</span>
    </div>
  )
}

function SourceList({ sources = [] }) {
  if (!sources.length) return <div className="source-empty">Подтверждённые источники не переданы.</div>
  return (
    <ul className="sources">
      {sources.map((source, index) => (
        <li key={`${source.url}-${index}`}>
          <div className="source-main">
            <a href={source.url} target="_blank" rel="noreferrer noopener">
              <span>{source.title || 'Источник без названия'}</span><Icon name="external" size={14} />
            </a>
            <span className={`trust trust-${source.trust_level || 'средний'}`}>{source.trust_level || 'средний'}</span>
          </div>
          <span className="meta">
            {source.source_type || 'источник'} · {source.language || 'язык не указан'} · {source.date || 'дата не указана'}
            {source.translated ? ' · автоперевод / русское резюме' : ''}
          </span>
        </li>
      ))}
    </ul>
  )
}

function Stats({ stats }) {
  if (!stats) return null
  const connectors = Object.entries(stats.connector_status || {})
  const online = connectors.filter(([, value]) => String(value).startsWith('ok')).length
  const cards = [
    ['Источники', stats.sources_processed, 'обработано', 'database'],
    ['Кандидаты', stats.candidates_total, 'до фильтрации', 'spark'],
    ['Отклонено', stats.candidates_rejected, 'зрелость / trust / шум', 'filter'],
    ['Уверенные', stats.confident_signals, 'уверенность > 75%', 'check'],
  ]
  return (
    <section className="stats-grid" aria-label="Статистика поиска">
      {cards.map(([label, value, sub, icon]) => (
        <div className="stat-card" key={label}>
          <div className="stat-icon"><Icon name={icon} /></div>
          <div className="stat-copy"><span>{label}</span><strong>{value ?? '—'}</strong><small>{sub}</small></div>
        </div>
      ))}
      <div className="stat-card model-card">
        <div className="stat-icon"><Icon name="spark" /></div>
        <div className="stat-copy"><span>Модель</span><strong className="model-name">{stats.llm_model || 'не указана'}</strong><small>{stats.ml_mode || 'ML'} · {stats.retrieval_mode || 'live-поиск'}</small></div>
      </div>
      <div className="stat-card model-card">
        <div className="stat-icon"><Icon name="database" /></div>
        <div className="stat-copy"><span>Источники-коннекторы</span><strong className="model-name">{online} / {connectors.length || 0}</strong><small>доступны в текущем прогоне</small></div>
      </div>
    </section>
  )
}

function SignalCard({ doc, index, onOpen }) {
  return (
    <article className="signal-card">
      <button type="button" className="signal-summary" onClick={onOpen} aria-label={`Открыть отчёт: ${doc.title}`}>
        <div className="rank">{String(index + 1).padStart(2, '0')}</div>
        <div className="signal-content">
          <div className="signal-topline">
            <span className="area">{doc.area || 'Технологический тренд'}</span>
            <span className="status-pill">{doc.is_weak_signal ? 'Слабый сигнал' : 'Проверка'}</span>
          </div>
          <h3>{doc.title}</h3>
          <p className="companies">{doc.companies?.length ? doc.companies.join(' · ') : 'Компании не определены по источникам'}</p>
          <div className="signal-tags">
            {doc.stage ? <span>{STAGE_LABELS[doc.stage] || `Стадия ${doc.stage}`}</span> : null}
            {doc.trend ? <span className="trend"><i />{TREND_LABELS[doc.trend] || 'динамика'}</span> : null}
            {doc.sources?.length ? <span>{doc.sources.length} {doc.sources.length === 1 ? 'источник' : 'источников'}</span> : null}
            {doc.dataset_score != null ? <span>датасет {doc.dataset_score}/7</span> : null}
          </div>
          <div className="signal-reason">
            <div><b>Почему слабый сигнал</b><p>{doc.why || 'Объяснение появится после оценки модели.'}</p></div>
            <div><b>Почему релевантен</b><p>{doc.relevance_reason || 'Связь с запросом подтверждена контекстным отсевом.'}</p></div>
            {doc.signal_markers?.length ? <div className="signal-markers">{doc.signal_markers.slice(0, 5).map((marker, markerIndex) => <span key={`${marker}-${markerIndex}`}>{marker}</span>)}</div> : null}
          </div>
        </div>
        <div className="signal-score"><Score value={doc.score} /><span className="expand-icon"><Icon name="arrow" size={17} /></span></div>
      </button>
      {doc.sources?.length ? (
        <div className="signal-sources-preview">
          <div className="signal-sources-head"><span>ИСТОЧНИКИ / ДОКАЗАТЕЛЬСТВА</span><b>{doc.sources.length}</b></div>
          <div className="signal-source-grid">
            {doc.sources.slice(0, 4).map((source, sourceIndex) => (
              <a key={`${source.url}-${sourceIndex}`} href={source.url} target="_blank" rel="noreferrer noopener">
                <span className="signal-source-title">{source.title || 'Источник без названия'}</span>
                <span className={`trust trust-${source.trust_level || 'средний'}`}>{source.trust_level || 'средний'}</span>
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  )
}

function SignalReport({ doc, onBack }) {
  const sourceCount = doc.sources?.length || 0
  return (
    <section className="report-view">
      <button type="button" className="report-back" onClick={onBack}>← Вернуться к ТОП-15</button>
      <div className="report-header">
        <div>
          <div className="eyebrow"><span /> ИССЛЕДОВАНИЕ / ДОКУМЕНТ</div>
          <div className="report-kicker">{doc.area || 'Технологический тренд'} · {doc.is_weak_signal ? 'Слабый сигнал' : 'Кандидат'}</div>
          <h2>{doc.title}</h2>
          <p className="report-summary">{doc.description || doc.raw_text || 'Описание отсутствует.'}</p>
        </div>
        <div className="report-score"><Score value={doc.score} /></div>
      </div>

      <div className="report-grid">
        <div className="report-block report-block-primary"><div className="detail-label">Почему это слабый сигнал</div><p>{doc.why || 'Объяснение классификации не передано.'}</p>{doc.signal_markers?.length ? <div className="report-markers">{doc.signal_markers.map((marker, index) => <span key={`${marker}-${index}`}>{marker}</span>)}</div> : null}</div>
        <div className="report-block"><div className="detail-label">Потенциальное преимущество</div><p>{doc.advantage || 'В подтверждённом контексте преимущество не выделено.'}</p></div>
        <div className="report-block"><div className="detail-label">Кейс-пример</div><p>{doc.case_example || 'Конкретный кейс в найденных источниках не указан.'}</p></div>
        <div className="report-block"><div className="detail-label">Почему релевантен запросу</div><p>{doc.relevance_reason || 'Релевантность подтверждена контекстным отсевом.'}</p></div><div className="report-block"><div className="detail-label">Предикторы и доказательства</div><p>{doc.evidence_summary || 'Основание сформировано по найденным источникам.'}</p><p className="source-agreement">{doc.source_agreement || `Источников в карточке: ${sourceCount}.`}</p></div>
      </div>

      <div className="report-meta-grid">
        <div><span>ML-режим</span><strong>{modelModeLabel(doc.model_mode)}</strong></div>
        <div><span>Версия модели</span><strong>{doc.model_version || 'logreg-l2-v4'}</strong></div>
        <div><span>Оценка датасета</span><strong>{doc.dataset_score != null ? `${doc.dataset_score}/7` : '—'}</strong></div>
        <div><span>Источники</span><strong>{sourceCount}</strong></div>
      </div>

      <div className="report-sources">
        <div className="detail-label">Проверяемые источники <span>{sourceCount}</span></div>
        <SourceList sources={doc.sources || []} />
      </div>
    </section>
  )
}

function ResearchPanel() {
  return (
    <aside className="research-panel" aria-label="Как формируется исследование">
      <div className="panel-overline">МЕТОД / 01—03</div>
      <div className="research-rule" />
      <div className="research-row"><span>01</span><div><strong>Открытые источники</strong><small>научные публикации · реестры · Crossref</small></div></div>
      <div className="research-row"><span>02</span><div><strong>Контекстный отсев</strong><small>тема документа должна совпасть с темой запроса</small></div></div>
      <div className="research-row"><span>03</span><div><strong>Объяснимый рейтинг</strong><small>уверенность · стадия · динамика · фактура</small></div></div>
      <div className="research-stamp"><b>OPEN</b><span>evidence first</span></div>
    </aside>
  )
}

function EmptyState({ onExample }) {
  return <div className="empty-state"><div className="empty-state-copy"><div className="empty-kicker"><span className="mini-pulse" /> ГОТОВ К ИССЛЕДОВАНИЮ</div><div className="empty-orbit"><Icon name="orbit" size={31} /></div><h2>Ищите то, что только начинает появляться</h2><p>Введите технологическую тему. Система соберёт открытые источники, проверит их доверенность, отфильтрует зрелые тренды и покажет TOP-15 ранних сигналов с объяснением.</p><div className="examples"><span>Быстрый старт</span>{EXAMPLES.map((example) => <button key={example} type="button" onClick={() => onExample(example)}>{example}</button>)}</div></div><div className="empty-state-aside"><div className="aside-line"><span>01</span><p>Соберём открытые источники</p><b>→</b></div><div className="aside-line"><span>02</span><p>Проверим доверенность и зрелость</p><b>→</b></div><div className="aside-line"><span>03</span><p>Сформируем объяснимый TOP-15</p><b>→</b></div></div></div>
}

function SearchProgress({ status, busy, elapsedMs }) {
  const seconds = Math.floor(elapsedMs / 1000)
  const stage = seconds < 12 ? 'Сбор открытых источников' : seconds < 55 ? 'Семантический отбор кандидатов' : 'Ранжирование, проверка и сбор доказательств'
  return <div className="progress-panel"><div className="status-bar"><div className="status-pulse"><i /></div><div className="progress-copy"><strong>{status || stage}</strong><span>{stage} · {seconds} сек.</span></div><div className="status-steps"><span className={seconds >= 0 ? 'active' : ''}>СБОР</span><i /><span className={seconds >= 12 ? 'active' : ''}>ОТСЕВ</span><i /><span className={seconds >= 55 ? 'active' : ''}>ОТЧЁТ</span></div></div>{busy ? <div className="progress-track"><i /></div> : null}</div>
}

export default function App() {
  const [query, setQuery] = useState('технологии для промышленного ИИ')
  const [status, setStatus] = useState('')
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [report, setReport] = useState(null)
  const [showRejected, setShowRejected] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)

  const visibleResults = useMemo(() => {
    // Сервер уже выполнил обязательный context gate и ML ranking.
    // Не фильтруем выдачу второй раз на клиенте: иначе хороший серверный TOP-15
    // мог превращаться в TOP-4/5 из-за более грубого браузерного фильтра.
    const results = Array.isArray(job?.results) ? job.results : []
    return results
      .slice(0, 15)
  }, [job])

  async function onSubmit(event) {
    event.preventDefault()
    const cleanQuery = query.trim()
    if (cleanQuery.length < 2 || busy) return

    setBusy(true); setError(''); setJob(null); setReport(null); setShowRejected(false); setElapsedMs(0)
    setStatus('Запускаем live-поиск…')
    const started = Date.now()

    try {
      const accepted = await startSearch(cleanQuery, 15)
      setStatus(`Поиск запущен · задача ${accepted.job_id.slice(0, 8)}…`)
      const done = await pollUntilDone(accepted.job_id, {
        timeoutMs: 180000,
        onProgress: (current, elapsed, transientError) => {
          setElapsedMs(elapsed)
          if (transientError) setStatus('Связь с API временно прервалась — продолжаем проверять задачу…')
          else if (current?.status === 'выполняется') setStatus('Поиск, фильтрация и ранжирование выполняются…')
        },
      })
      setElapsedMs(Date.now() - started)
      if (done.status === 'ошибка') {
        setError(done.error || 'Поиск завершился с ошибкой.')
        setStatus('Поиск завершился с ошибкой')
      } else {
        setJob(done)
        setStatus(`Готово · сформировано ${done.results?.length || 0} сигналов`)
      }
    } catch (e) {
      setError(e.message || 'Не удалось выполнить поиск.')
      setStatus('Поиск не завершён')
    } finally {
      setBusy(false)
    }
  }

  async function openReport(doc) {
    setError('')
    // Критично: score из выдачи является снимком конкретного прогона.
    // /signal/{id} может повторно обогащать документ и пересчитать его — поэтому
    // при открытии карточки никогда не заменяем исходный score новым значением.
    const snapshot = { ...doc, score: stableScore(doc.score) }
    setReport(snapshot)
    window.scrollTo({ top: 0, behavior: 'smooth' })

    try {
      const full = await fetchSignal(doc.id)
      setReport((current) => current ? {
        ...current,
        ...full,
        id: snapshot.id,
        title: snapshot.title,
        score: snapshot.score,
        dataset_score: snapshot.dataset_score,
        relevance_reason: full.relevance_reason || snapshot.relevance_reason,
        signal_markers: full.signal_markers?.length ? full.signal_markers : snapshot.signal_markers,
        source_agreement: full.source_agreement || snapshot.source_agreement,
      } : current)
    } catch (e) {
      // Карточка уже полностью открыта из результата поиска; enrichment необязателен.
      setStatus('Отчёт открыт · исходный score зафиксирован')
    }
  }

  function setExample(value) { setQuery(value); setJob(null); setReport(null); setError(''); setStatus('') }

  return <div className="app-shell">
    <div className="ambient ambient-one" /><div className="ambient ambient-two" /><div className="ambient ambient-three" />
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Icon name="spark" size={20} /></div><div><div className="brand-name">WEAK<span>0x00</span></div><div className="brand-sub">RESEARCH / EARLY SIGNALS</div></div></div>
      <div className="topbar-meta"><span className="topbar-meta-item"><i className="live-dot" /> OPEN SOURCES</span><span className="topbar-separator">/</span><span className="topbar-meta-item">EVIDENCE FIRST</span></div>
      <div className="live-badge"><i /> LIVE RESEARCH</div>
    </header>

    <main>
      <section className="hero">
        <div className="hero-content"><div className="eyebrow"><span /> РАННИЕ ТЕХНОЛОГИЧЕСКИЕ ТРЕНДЫ</div><h1>Найдите тему<br /><em>до того, как она станет очевидной.</em></h1><p className="hero-copy">Исследуйте открытые источники и собирайте ранние технологические сигналы. Каждый результат остаётся привязан к исходным материалам, а нерелевантные документы отсеиваются до рейтинга.</p>
          <form onSubmit={onSubmit} className="search-box"><div className="search-icon"><Icon name="search" /></div><input type="text" value={query} placeholder="Например: новые технологии защиты ИИ-моделей" onChange={(e) => setQuery(e.target.value)} aria-label="Поисковый запрос" maxLength={500} /><button type="submit" disabled={busy || query.trim().length < 2}>{busy ? <><span className="spinner" /> Ищем</> : <>Исследовать <Icon name="arrow" size={17} /></>}</button></form>
          <div className="search-note"><span>LIVE / открытые источники</span><span>TOP-15 сигналов</span><span>контекстный отсев</span></div>
        </div>
        <ResearchPanel />
      </section>

      {(status || busy) && <SearchProgress status={status} busy={busy} elapsedMs={elapsedMs} />}
      {error && <div className="error-bar"><strong>Внимание</strong><span>{error}</span><button type="button" onClick={() => setError('')} aria-label="Закрыть"><Icon name="close" size={15} /></button></div>}

      {!job && !busy && !error && <EmptyState onExample={setExample} />}

      {job && <section className="results-section">
        <div className="section-heading"><div><div className="eyebrow"><span /> РЕЗУЛЬТАТ ИССЛЕДОВАНИЯ</div><h2>Сигналы по запросу <span>«{job.query}»</span></h2></div><div className="result-count"><strong>{visibleResults.length}</strong><small> / 15 релевантных</small></div></div>
        <Stats stats={job.stats} />
        {job.results?.length > visibleResults.length ? <div className="relevance-note"><span>RELEVANCE GATE</span><p>Серверный контекстный отсев выполнен до ML-ранжирования; ниже показывается исходный TOP-15 без дополнительного клиентского отсечения.</p></div> : null}
        {!report ? <>
          {visibleResults.length ? <><div className="list-head"><div><span>РАНГ</span><span>ГИПОТЕЗА</span></div><span>УВЕРЕННОСТЬ</span></div><div className="signals-list">{visibleResults.map((doc, i) => <SignalCard key={doc.id || `${doc.title}-${i}`} doc={doc} index={i} onOpen={() => openReport(doc)} />)}</div></> : <div className="no-results"><div className="empty-orbit"><Icon name="search" size={26} /></div><h3>Подходящих сигналов не найдено</h3><p>Попробуйте расширить технологический запрос. Система не добавляет неподтверждённые кандидаты только ради заполнения списка.</p></div>}
          {job.rejected?.length ? <div className="rejected-block"><button type="button" className="rejected-toggle" onClick={() => setShowRejected(!showRejected)}><span className="toggle-chevron">{showRejected ? '−' : '+'}</span><span>Отклонённые кандидаты</span><b>{job.rejected.length}</b><small>{showRejected ? 'скрыть причины' : 'показать причины'}</small></button>{showRejected && <div className="rejected-list">{job.rejected.map((doc, index) => <div className="rejected-item" key={doc.id || index}><div><strong>{doc.title}</strong><span>{doc.rejected_reason || 'Не прошёл итоговый фильтр.'}</span></div></div>)}</div>}</div> : null}
        </> : <SignalReport doc={report} onBack={() => { setReport(null); setError('') }} />}
      </section>}
    </main>

    <footer><span>WEAK0x00 SIGNALS</span><span>Доказательный поиск по открытым источникам</span><span>© / EARLY SIGNALS SYSTEM</span></footer>
  </div>
}
