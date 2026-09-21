// Единственная точка обращения к бэкенду. Только относительные пути /api — их проксирует Caddy,
// поэтому фронту не нужно знать ни адреса api, ни домена.

const BASE = '/api'

async function request(path, options) {
  const resp = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    ...options,
  })
  if (!resp.ok) {
    let detail = 'Ошибка ' + resp.status
    try {
      const body = await resp.json()
      if (body && body.detail) detail = body.detail
    } catch (e) {
      /* тело не JSON — оставляем код ошибки */
    }
    throw new Error(detail)
  }
  return resp.json()
}

export function startSearch(query, limit = 15) {
  return request('/search', {
    method: 'POST',
    body: JSON.stringify({ query, limit }),
  })
}

export function fetchResult(jobId) {
  return request('/search/' + encodeURIComponent(jobId))
}

export function fetchSignal(signalId) {
  return request('/signal/' + encodeURIComponent(signalId))
}

export function fetchSources(limit = 100) {
  return request('/sources?limit=' + limit)
}

// Опрос задачи до завершения (job_id + polling).
export async function pollUntilDone(jobId, { intervalMs = 1000, timeoutMs = 120000 } = {}) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const job = await fetchResult(jobId)
    if (job.status === 'готово' || job.status === 'ошибка') return job
    await new Promise((r) => setTimeout(r, intervalMs))
  }
  throw new Error('Превышено время ожидания результата поиска.')
}
