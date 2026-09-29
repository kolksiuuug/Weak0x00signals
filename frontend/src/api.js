// Единый API-клиент фронтенда. Все запросы идут через относительный /api,
// поэтому приложение одинаково работает локально и за Caddy.

const BASE = '/api'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function request(path, options = {}, { retries = 2, retryDelay = 700 } = {}) {
  let lastError
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetch(BASE + path, {
        ...options,
        headers: {
          Accept: 'application/json',
          ...(options.body ? { 'Content-Type': 'application/json' } : {}),
          ...(options.headers || {}),
        },
      })

      if (!response.ok) {
        let detail = `Ошибка ${response.status}`
        try {
          const body = await response.json()
          if (body?.detail) detail = body.detail
        } catch (_) { /* non-json body */ }

        // 404/400/422 — это уже содержательная ошибка запроса, повторять её бессмысленно.
        if (response.status < 500 && response.status !== 429) throw new Error(detail)
        lastError = new Error(detail)
      } else {
        return await response.json()
      }
    } catch (error) {
      lastError = error
      // Сетевые ошибки и 5xx повторяем: контейнеры могут несколько секунд стартовать.
      if (error instanceof TypeError || /Ошибка 5\d\d|Ошибка 429|недоступ/i.test(error.message)) {
        // retry below
      } else {
        throw error
      }
    }

    if (attempt < retries) await sleep(retryDelay * (attempt + 1))
  }
  throw lastError || new Error('API недоступен')
}

export function startSearch(query, limit = 15) {
  return request('/search', {
    method: 'POST',
    body: JSON.stringify({ query, limit }),
  }, { retries: 3, retryDelay: 900 })
}

export function fetchResult(jobId) {
  return request('/search/' + encodeURIComponent(jobId), {}, { retries: 3, retryDelay: 800 })
}

export function fetchSignal(signalId) {
  return request('/signal/' + encodeURIComponent(signalId), {}, { retries: 2, retryDelay: 800 })
}

export function fetchSources(limit = 100) {
  return request('/sources?limit=' + limit)
}

export async function pollUntilDone(
  jobId,
  {
    intervalMs = 900,
    timeoutMs = 180000,
    onProgress,
  } = {},
) {
  const startedAt = Date.now()
  const deadline = startedAt + timeoutMs
  let transientFailures = 0
  let lastJob = null

  while (Date.now() < deadline) {
    try {
      const job = await fetchResult(jobId)
      lastJob = job
      transientFailures = 0
      onProgress?.(job, Date.now() - startedAt)
      if (job.status === 'готово' || job.status === 'ошибка') return job
    } catch (error) {
      transientFailures += 1
      // Один-два неудачных GET не должны ломать уже выполняющуюся задачу.
      if (transientFailures >= 5) throw error
      onProgress?.(lastJob, Date.now() - startedAt, error)
    }
    await sleep(intervalMs)
  }

  throw new Error('Поиск выполняется дольше 3 минут. Задача продолжает работать в фоне — повторите проверку результата.')
}
