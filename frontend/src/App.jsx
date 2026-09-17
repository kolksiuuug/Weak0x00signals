import { useState } from 'react'
import { startSearch, pollUntilDone } from './api.js'

const STAGE_LABELS = {
  1: 'Концепция/Исследование',
  2: 'Прототип/PoC',
  3: 'Пилот',
  4: 'Раннее внедрение',
}
const TREND_LABELS = { 1: 'стабильно', 2: 'растёт', 3: 'растёт быстро' }

function Stats({ stats }) {
  if (!stats) return null
  return (
    <div className="stats">
      <div><b>{stats.sources_processed}</b><span>обработано источников</span></div>
      <div><b>{stats.candidates_total}</b><span>кандидатов найдено</span></div>
      <div><b>{stats.candidates_rejected}</b><span>отклонено фильтрами</span></div>
      <div><b>{stats.confident_signals}</b><span>сигналов с уверенностью &gt; 75 %</span></div>
      <div><b>{stats.llm_model || '—'}</b><span>используемая LLM</span></div>
    </div>
  )
}

function SourceList({ sources }) {
  return (
    <ul className="sources">
      {sources.map((s, i) => (
        <li key={i}>
          <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
          <span className={'trust trust-' + s.trust_level}>{s.trust_level}</span>
          <span className="meta">
            {s.source_type} · {s.language} · {s.date || 'дата не указана'}
            {s.translated ? ' · автоперевод/генеративное резюме' : ''}
          </span>
        </li>
      ))}
    </ul>
  )
}

function SignalRow({ doc, index, expanded, onToggle }) {
  const percent = doc.score != null ? Math.round(doc.score * 100) : null
  return (
    <>
      <tr className={expanded ? 'row expanded' : 'row'} onClick={onToggle}>
        <td className="num">{index + 1}</td>
        <td>
          <div className="title">{doc.title}</div>
          <div className="companies">{doc.companies.join(', ') || 'компании не определены'}</div>
        </td>
        <td>{doc.area || '—'}</td>
        <td className="score">{percent != null ? percent + ' %' : '—'}</td>
        <td>{STAGE_LABELS[doc.stage] || '—'}</td>
        <td>{TREND_LABELS[doc.trend] || '—'}</td>
        <td>{doc.is_weak_signal ? 'слабый сигнал' : 'под вопросом'}</td>
      </tr>
      {expanded && (
        <tr className="card">
          <td colSpan={7}>
            <h4>Почему это слабый сигнал</h4>
            <p>{doc.why || 'Объяснение появится после подключения модели.'}</p>
            <h4>Описание</h4>
            <p>{doc.raw_text}</p>
            <h4>Источники ({doc.sources.length})</h4>
            <SourceList sources={doc.sources} />
          </td>
        </tr>
      )}
    </>
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
    setBusy(true)
    setError('')
    setJob(null)
    setOpenId(null)
    setStatus('Запрос поставлен в очередь…')
    try {
      const accepted = await startSearch(query.trim())
      setStatus('Идёт сбор источников и скоринг (задача ' + accepted.job_id.slice(0, 8) + ')…')
      const done = await pollUntilDone(accepted.job_id)
      if (done.status === 'ошибка') {
        setError(done.error || 'Задача завершилась с ошибкой.')
        setStatus('')
      } else {
        setJob(done)
        setStatus('Готово: найдено гипотез — ' + done.results.length)
      }
    } catch (e) {
      setError(e.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Слабые сигналы</h1>
        <p className="lead">
          Поиск ранних научно-технологических трендов по открытым источникам.
          Скелет работает на МОК-данных — реальные коннекторы и модель подключаются отдельно.
        </p>
      </header>

      <form onSubmit={onSubmit} className="search">
        <input
          type="text"
          value={query}
          placeholder="Например: новые технологии защиты ИИ-моделей"
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={busy}>{busy ? 'Ищем…' : 'Найти'}</button>
      </form>

      {status && <div className="status">{status}</div>}
      {error && <div className="error">Ошибка: {error}</div>}

      {job && (
        <>
          <Stats stats={job.stats} />

          <h2>ТОП-{job.results.length} гипотез по запросу «{job.query}»</h2>
          <table className="results">
            <thead>
              <tr>
                <th>#</th>
                <th>Технология</th>
                <th>Область</th>
                <th>Скоринг</th>
                <th>Стадия</th>
                <th>Тренд</th>
                <th>Статус</th>
              </tr>
            </thead>
            <tbody>
              {job.results.map((doc, i) => (
                <SignalRow
                  key={doc.id}
                  doc={doc}
                  index={i}
                  expanded={openId === doc.id}
                  onToggle={() => setOpenId(openId === doc.id ? null : doc.id)}
                />
              ))}
            </tbody>
          </table>
          <p className="hint">Нажмите на строку, чтобы раскрыть карточку-инсайт с источниками.</p>

          <h2 className="rejected-header" onClick={() => setShowRejected(!showRejected)}>
            {showRejected ? '▾' : '▸'} Отклонённые кандидаты ({job.rejected.length}) — с причиной по каждому
          </h2>
          {showRejected && (
            <ul className="rejected">
              {job.rejected.map((doc) => (
                <li key={doc.id}>
                  <b>{doc.title}</b>
                  <span>{doc.rejected_reason}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      <footer>
        Хакатон Газпромбанк.Тех · выдача формируется только по проверенным источникам
      </footer>
    </div>
  )
}
