import { useState } from 'react'
import { startSearch, pollUntilDone } from './api.js'

const STAGE_LABELS = { 1: 'Концепция / исследование', 2: 'Прототип / PoC', 3: 'Пилот', 4: 'Раннее внедрение' }
const TREND_LABELS = { 1: 'стабильно', 2: 'растёт', 3: 'растёт быстро' }

function Icon({ name, size = 18 }) {
  const paths = {
    search: <><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4 4" /></>,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" /></>,
    chevron: <path d="m7 10 5 5 5-5" />,
    external: <><path d="M14 5h5v5" /><path d="m19 5-8 8" /><path d="M19 13v5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5" /></>,
    activity: <><path d="M3 12h4l2-7 4 14 2-7h6" /></>,
    database: <><ellipse cx="12" cy="5" rx="7" ry="3" /><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5" /><path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="8" /><path d="M12 8v5l3 2" /></>,
    orbit: <><circle cx="12" cy="12" r="7" /><circle cx="12" cy="12" r="2.2" /><path d="M5 6.8C7.2 4.4 10.2 3 13.2 3c4.1 0 7.5 2.7 7.8 6.4" /></>,
  }
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}

const humanTrust = { высокий: 'высокий', средний: 'средний', пониженный: 'пониженный' }

function SignalField() {
  return (
    <div className="signal-field" aria-hidden="true">
      <div className="signal-field-glow" />
      <div className="signal-grid" />
      <div className="signal-orbit orbit-a" />
      <div className="signal-orbit orbit-b" />
      <div className="signal-core"><span /><i /><b>LIVE</b></div>
      <span className="signal-node node-a" />
      <span className="signal-node node-b" />
      <span className="signal-node node-c" />
      <span className="signal-node node-d" />
      <span className="signal-node node-e" />
      <span className="signal-path path-a" />
      <span className="signal-path path-b" />
      <span className="signal-path path-c" />
      <div className="signal-readout readout-a"><span>EARLY</span><strong>0.82</strong></div>
      <div className="signal-readout readout-b"><span>NOISE</span><strong>LOW</strong></div>
      <div className="signal-readout readout-c"><span>VECTOR</span><strong>RISING</strong></div>
    </div>
  )
}

function Stats({ stats }) {
  if (!stats) return null
  const items = [
    ['Источники', stats.sources_processed, 'обработано', 'database'],
    ['Кандидаты', stats.candidates_total, 'найдено', 'spark'],
    ['Фильтры', stats.candidates_rejected, 'отклонено', 'activity'],
    ['Уверенные', stats.confident_signals, 'уверенность > 75%', 'check'],
  ]
  const connectorCount = Object.values(stats.connector_status || {}).filter((v) => String(v).startsWith('ok')).length
  return (
    <section className="stats-grid">
      {items.map(([label, value, sub, icon]) => (
        <div className="stat-card" key={label}>
          <div className="stat-icon"><Icon name={icon} /></div>
          <div className="stat-copy"><span>{label}</span><strong>{value ?? '—'}</strong><small>{sub}</small></div>
        </div>
      ))}
      <div className="stat-card model-card">
        <div className="stat-icon"><Icon name="spark" /></div>
        <div className="stat-copy">
          <span>Контур обработки</span>
          <strong className="model-name">{stats.llm_model || 'LLM не настроена'}</strong>
          <small>{stats.ml_mode || 'эвристика'} · {stats.retrieval_mode === 'live' ? 'live RAG' : stats.retrieval_mode || 'поиск'}</small>
        </div>
      </div>
      <div className="stat-card model-card">
        <div className="stat-icon"><Icon name="database" /></div>
        <div className="stat-copy">
          <span>Коннекторы</span>
          <strong className="model-name">{connectorCount} / {Object.keys(stats.connector_status || {}).length || 0}</strong>
          <small>доступны в текущем прогоне</small>
        </div>
      </div>
    </section>
  )
}

function Score({ value }) {
  if (value == null) return <span className="score muted-score">—</span>
  const percent = Math.round(value * 100)
  return (
    <div className="score-wrap">
      <div className="score-ring" style={{ '--score': `${percent * 3.6}deg` }}>
        <span>{percent}</span>
      </div>
      <span className="score-caption">уверенность</span>
    </div>
  )
}

function SourceList({ sources }) {
  return (
    <ul className="sources">
      {sources.map((s, i) => (
        <li key={`${s.url}-${i}`}>
          <div className="source-main">
            <a href={s.url} target="_blank" rel="noreferrer">{s.title}<Icon name="external" size={14} /></a>
            <span className={'trust trust-' + s.trust_level}>{humanTrust[s.trust_level] || s.trust_level}</span>
          </div>
          <span className="meta">{s.source_type} · {s.language} · {s.date || 'дата не указана'}{s.translated ? ' · автоперевод / резюме' : ''}</span>
        </li>
      ))}
    </ul>
  )
}

function SignalCard({ doc, index, expanded, onToggle }) {
  const stage = STAGE_LABELS[doc.stage] || '—'
  const trend = TREND_LABELS[doc.trend] || '—'
  return (
    <article className={'signal-card' + (expanded ? ' is-open' : '')}>
      <button className="signal-summary" onClick={onToggle} aria-expanded={expanded}>
        <div className="rank">{String(index + 1).padStart(2, '0')}</div>
        <div className="signal-content">
          <div className="signal-topline"><span className="area">{doc.area || 'Технологический тренд'}</span><span className="status-pill">{doc.is_weak_signal ? 'Слабый сигнал' : 'Проверка'}</span></div>
          <h3>{doc.title}</h3>
          <p className="companies">{doc.companies?.length ? doc.companies.join(' · ') : 'Компании не определены по источникам'}</p>
          <div className="signal-tags"><span>{stage}</span><span className="trend"><i />{trend}</span>{doc.dataset_score ? <span>балл {doc.dataset_score}/7</span> : null}</div>
        </div>
        <div className="signal-score"><Score value={doc.score} /><span className="expand-icon"><Icon name="chevron" /></span></div>
      </button>
      {expanded && (
        <div className="signal-details">
          <div className="detail-column"><div className="detail-label">Почему это слабый сигнал</div><p>{doc.why || 'Объяснение отсутствует.'}</p></div>
          <div className="detail-column"><div className="detail-label">Описание</div><p>{doc.description || doc.raw_text || 'Описание отсутствует.'}</p></div>
          <div className="detail-column"><div className="detail-label">Потенциальное преимущество</div><p>{doc.advantage || 'Верифицированного преимущества в найденном контексте не выделено.'}</p></div>
          <div className="detail-column"><div className="detail-label">Кейс-пример</div><p>{doc.case_example || 'Конкретный кейс в найденных источниках не указан.'}</p></div>
          <div className="detail-column"><div className="detail-label">Основание</div><p>{doc.evidence_summary || 'Основание сформировано по найденным источникам.'}</p></div>
          <div className="detail-column sources-column"><div className="detail-label">Источники <span>{doc.sources?.length || 0}</span></div><SourceList sources={doc.sources || []} /></div>
        </div>
      )}
    </article>
  )
}

function EmptyState({ onExample }) {
  return (
    <div className="empty-state">
      <div className="empty-state-copy">
        <div className="empty-kicker"><span className="mini-pulse" /> ПОИСКОВОЕ ПОЛЕ ГОТОВО</div>
        <div className="empty-orbit"><Icon name="orbit" size={31} /></div>
        <h2>Ищите то, что только начинает появляться</h2>
        <p>Введите технологическую тему — система соберёт открытые источники, отфильтрует зрелые и шумовые технологии и покажет ранние сигналы.</p>
        <div className="examples"><span>Можно начать с</span><button type="button" onClick={() => onExample('защита ИИ-моделей')}>защита ИИ-моделей</button><button type="button" onClick={() => onExample('роботы для промышленности')}>роботы для промышленности</button><button type="button" onClick={() => onExample('инфраструктура для ИИ')}>инфраструктура для ИИ</button></div>
      </div>
      <div className="empty-state-aside">
        <div className="aside-line"><span>01</span><p>Соберём открытые источники</p><b>→</b></div>
        <div className="aside-line"><span>02</span><p>Проверим зрелость и шум</p><b>→</b></div>
        <div className="aside-line"><span>03</span><p>Покажем слабые сигналы</p><b>×</b></div>
      </div>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('технологии для промышленного ИИ')
  const [status, setStatus] = useState('')
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [openId, setOpenId] = useState(null)
  const [showRejected, setShowRejected] = useState(false)

  async function onSubmit(event) {
    event.preventDefault()
    if (!query.trim() || busy) return
    setBusy(true); setError(''); setJob(null); setOpenId(null); setShowRejected(false)
    setStatus('Собираем открытые источники…')
    try {
      const accepted = await startSearch(query.trim())
      setStatus('Поиск и скоринг выполняются · задача ' + accepted.job_id.slice(0, 8) + '…')
      const done = await pollUntilDone(accepted.job_id)
      if (done.status === 'ошибка') { setError(done.error || 'Задача завершилась с ошибкой.'); setStatus('') }
      else { setJob(done); setStatus('Готово · найдено гипотез: ' + done.results.length) }
    } catch (e) { setError(e.message); setStatus('') } finally { setBusy(false) }
  }

  function setExample(value) { setQuery(value); setJob(null); setError(''); setStatus('') }

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" /><div className="ambient ambient-three" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Icon name="spark" size={20} /></div>
          <div>
            <div className="brand-name">WEAK<span>0x00</span></div>
            <div className="brand-sub">СЛАБЫЕ СИГНАЛЫ / ИССЛЕДОВАТЕЛЬСКАЯ КОНСОЛЬ</div>
          </div>
        </div>
        <div className="topbar-meta"><span className="topbar-meta-item"><i className="live-dot" /> OPEN SOURCES</span><span className="topbar-separator">/</span><span className="topbar-meta-item">RAG + ML</span></div>
        <div className="live-badge"><i /> СИСТЕМА ОНЛАЙН</div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow"><span /> РАННИЕ ТЕХНОЛОГИЧЕСКИЕ ТРЕНДЫ</div>
            <h1>Увидеть сигнал<br /><em>до того, как он станет трендом.</em></h1>
            <p className="hero-copy">Исследуйте открытые источники, находите малоочевидные технологические сдвиги и получайте гипотезы с объяснением и подтверждающими источниками.</p>
            <form onSubmit={onSubmit} className="search-box">
              <div className="search-icon"><Icon name="search" /></div>
              <input type="text" value={query} placeholder="Например: новые технологии защиты ИИ-моделей" onChange={(e) => setQuery(e.target.value)} aria-label="Поисковый запрос" />
              <button type="submit" disabled={busy}>{busy ? <><span className="spinner" /> Ищем</> : <>Исследовать <Icon name="arrow" size={17} /></>}</button>
            </form>
            <div className="search-note"><span><Icon name="clock" size={14} /> Поиск по открытым источникам</span><span>RAG · PostgreSQL · pgvector</span><span className="search-note-live"><i /> live pipeline</span></div>
          </div>
          <SignalField />
        </section>

        {status && <div className="status-bar"><div className="status-pulse"><i /></div><span>{status}</span><div className="status-steps"><span>COLLECT</span><i /> <span>SCORE</span><i /> <span>EXPLAIN</span></div></div>}
        {error && <div className="error-bar"><strong>Ошибка</strong><span>{error}</span></div>}

        {!job && !busy && !error && <EmptyState onExample={setExample} />}

        {job && (
          <section className="results-section">
            <div className="section-heading">
              <div>
                <div className="eyebrow"><span /> РЕЗУЛЬТАТ ИССЛЕДОВАНИЯ</div>
                <h2>Сигналы по запросу <span>«{job.query}»</span></h2>
              </div>
              <div className="result-count"><strong>{job.results.length}</strong><small> гипотез</small></div>
            </div>
            <Stats stats={job.stats} />
            <div className="list-head"><div><span>РАНГ</span><span>ГИПОТЕЗА</span></div><span>УВЕРЕННОСТЬ</span></div>
            <div className="signals-list">{job.results.map((doc, i) => <SignalCard key={doc.id} doc={doc} index={i} expanded={openId === doc.id} onToggle={() => setOpenId(openId === doc.id ? null : doc.id)} />)}</div>
            <p className="hint"><span>↳</span> Нажмите на гипотезу, чтобы открыть обоснование, кейс и источники.</p>
            <div className="rejected-block">
              <button className="rejected-toggle" onClick={() => setShowRejected(!showRejected)}>
                <span className="toggle-chevron">{showRejected ? '−' : '+'}</span>
                <span>Отклонённые кандидаты</span><b>{job.rejected.length}</b>
                <small>{showRejected ? 'скрыть' : 'показать причины'}</small>
              </button>
              {showRejected && <div className="rejected-list">{job.rejected.map((doc) => <div className="rejected-item" key={doc.id}><div><strong>{doc.title}</strong><span>{doc.rejected_reason}</span></div></div>)}</div>}
            </div>
          </section>
        )}
      </main>

      <footer><span>WEAK0x00 SIGNALS</span><span>Доказательный поиск по открытым источникам</span><span>© / EARLY SIGNALS SYSTEM</span></footer>
    </div>
  )
}
