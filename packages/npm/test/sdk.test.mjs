import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import test from 'node:test';
import { Sokudan, SokudanHTTPError } from '../dist/index.js';

const request = {
  state: '先月の請求が二重になっています',
  questions: {
    department: { type: 'choice', instructions: '担当部署', criteria: { 請求: null, 技術: '障害' } },
    urgency: { type: 'score', instructions: '緊急度', criteria: ['低', { label: '高' }] },
    churn: { type: 'bool', instructions: '解約したいか' },
  },
};
const response = {
  model: 'sokudan-ja-310m',
  answers: {
    department: { type: 'choice', choice: '請求', probabilities: { 請求: 0.9, 技術: 0.1 }, confidence: 0.8 },
    urgency: { type: 'score', score: 0.75, probabilities: { 0: 0.25, 1: 0.75 },
      legend: { 0: '低', 1: { label: '高' } }, confidence: 0.5 },
    churn: { type: 'noul', noul: 0.1 },
  },
  usage: { input_tokens: 60, output_tokens: 0 },
  sokudan: { state_format: 'text', state_tokens: 12, state_truncated: false,
    backbone_passes: 3, calibrated: true, calibrated_answers: ['churn'], latency_ms: 13 },
};

test('HTTP transport preserves request, all answer shapes, proxy prefix and bearer header', async (t) => {
  const calls = [];
  const server = createServer(async (req, res) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    calls.push({ url: req.url, method: req.method, headers: req.headers, body });
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(req.url.endsWith('/health') ? { status: 'ok' } : response));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const client = new Sokudan({ baseURL: `http://127.0.0.1:${server.address().port}/proxy`, apiKey: 'test' });
  assert.deepEqual(await client.predict(request), response);
  assert.deepEqual(await client.health(), { status: 'ok' });
  assert.equal(calls[0].url, '/proxy/v1/systemone');
  assert.equal(calls[0].method, 'POST');
  assert.equal(calls[0].headers.authorization, 'Bearer test');
  assert.equal(calls[0].headers['content-type'], 'application/json');
  assert.deepEqual(JSON.parse(calls[0].body), request);
  assert.equal(calls[1].url, '/proxy/health');
  assert.equal(calls[1].method, 'GET');
  assert.equal(calls[1].body, '');
});

for (const status of [413, 422, 429, 503]) {
  test(`HTTP ${status} preserves detail and Retry-After without retries`, async () => {
    let calls = 0;
    const body = { detail: [{ loc: ['body', 'state'], msg: 'test error' }] };
    const client = new Sokudan({ fetch: async () => {
      calls++;
      return new Response(JSON.stringify(body), { status, headers: { 'Retry-After': '1' } });
    } });
    await assert.rejects(client.predict(request), (error) => {
      assert.ok(error instanceof SokudanHTTPError);
      assert.equal(error.status, status);
      assert.deepEqual(error.body, body);
      assert.equal(error.headers.get('retry-after'), '1');
      return true;
    });
    assert.equal(calls, 1);
  });
}

test('non-JSON proxy errors retain HTTP status and body', async () => {
  const client = new Sokudan({ fetch: async () => new Response('Bad Gateway', { status: 502 }) });
  await assert.rejects(client.health(), (error) => error.status === 502 && error.body === 'Bad Gateway');
});

test('malformed successful JSON is not presented as a typed response', async () => {
  const client = new Sokudan({ fetch: async () => new Response('not json') });
  await assert.rejects(client.health(), SyntaxError);
});

test('network errors are preserved', async () => {
  const failure = new TypeError('connection refused');
  const client = new Sokudan({ fetch: async () => { throw failure; } });
  await assert.rejects(client.health(), (error) => error === failure);
});

function waitingFetch(_url, { signal }) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
}

test('deadline aborts a stalled request', async () => {
  const client = new Sokudan({ timeoutMs: 20, fetch: waitingFetch });
  await assert.rejects(client.health(), { name: 'TimeoutError' });
});

test('caller cancellation aborts an in-flight request', async () => {
  const controller = new AbortController();
  const client = new Sokudan({ fetch: waitingFetch });
  const pending = client.health({ signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
});

test('already-aborted signals remain aborted', async () => {
  const client = new Sokudan({ fetch: waitingFetch });
  await assert.rejects(client.health({ signal: AbortSignal.abort() }), { name: 'AbortError' });
});

test('deadline includes reading a stalled response body', async () => {
  const client = new Sokudan({ timeoutMs: 20, fetch: async (_url, { signal }) => ({
    ok: true,
    text: () => waitingFetch(null, { signal }),
  }) });
  await assert.rejects(client.health(), { name: 'TimeoutError' });
});

test('configuration rejects ambiguous URLs and invalid timeouts', () => {
  for (const baseURL of ['file:///tmp/server', 'http://localhost/?q=1', 'http://localhost/#x']) {
    assert.throws(() => new Sokudan({ baseURL }), TypeError);
  }
  for (const timeoutMs of [0, -1, Infinity, NaN, 1.2, 2_147_483_648]) {
    assert.throws(() => new Sokudan({ timeoutMs }), TypeError);
  }
});
