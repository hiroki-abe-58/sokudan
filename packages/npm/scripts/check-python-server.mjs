import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:net';
import { setTimeout } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { Sokudan, SokudanHTTPError } from '../dist/index.js';

const python = process.env.SOKUDAN_TEST_PYTHON;
assert.ok(python, 'set SOKUDAN_TEST_PYTHON to Python with sokudan[serve] and pytest installed');
const reservation = createServer();
reservation.listen(0, '127.0.0.1');
await once(reservation, 'listening');
const { port } = reservation.address();
await new Promise((resolve) => reservation.close(resolve));
const child = spawn(python, [fileURLToPath(new URL('../test/python-server.py', import.meta.url)), String(port)], {
  stdio: ['ignore', 'pipe', 'pipe'],
});
let log = '';
child.stdout.on('data', (data) => { log += data; });
child.stderr.on('data', (data) => { log += data; });
let spawnError;
child.on('error', (error) => { spawnError = error; });
const closed = once(child, 'close');
const client = new Sokudan({ baseURL: `http://127.0.0.1:${port}`, timeoutMs: 1000 });
try {
  const deadline = Date.now() + 30_000;
  let ready = false;
  while (Date.now() < deadline) {
    if (spawnError) throw spawnError;
    assert.equal(child.exitCode, null, log);
    try { ready = (await client.health()).status === 'ok'; }
    catch (error) {
      if (!(error instanceof TypeError) && error.name !== 'TimeoutError') throw error;
    }
    if (ready) break;
    await setTimeout(100);
  }
  assert.ok(ready, `Python server did not become ready\n${log}`);
  const response = await client.predict({ state: '請求が二重です', questions: {
    department: { type: 'choice', instructions: '担当部署は', criteria: { 請求: null, 技術: '障害' } },
    urgency: { type: 'score', instructions: '緊急度は', criteria: ['低', { level: '高' }] },
    churn: { type: 'bool', instructions: '解約したいか' },
  } });
  assert.equal(response.answers.department.type, 'choice');
  assert.equal(response.answers.department.choice, '請求');
  assert.equal(response.answers.urgency.type, 'score');
  assert.deepEqual(response.answers.urgency.legend, { 0: '低', 1: { level: '高' } });
  assert.equal(response.answers.churn.type, 'noul');
  assert.equal(response.usage.output_tokens, 0);
  assert.equal(response.sokudan.backbone_passes, 3);
  await assert.rejects(client.predict({ state: null, questions: {} }), (error) =>
    error instanceof SokudanHTTPError && error.status === 422 && Array.isArray(error.body.detail));
  console.log('Python HTTP integration: choice, score, bool alias, health and 422 passed');
} finally {
  child.kill('SIGTERM');
  await closed;
}
