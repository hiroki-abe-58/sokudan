import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { launchCommand, version } from '../dist/launcher.mjs';

const cli = fileURLToPath(new URL('../dist/cli.mjs', import.meta.url));
function run(args, env = {}) {
  return spawnSync(process.execPath, [cli, ...args], {
    encoding: 'utf8', env: { ...process.env, ...env }, timeout: 10_000,
  });
}

test('help and version work without uv or Python', () => {
  const env = { PATH: '', SOKUDAN_PYTHON: '' };
  const help = run(['--help'], env);
  assert.equal(help.status, 0);
  assert.match(help.stdout, /serve/);
  const result = run(['--version'], env);
  assert.equal(result.status, 0);
  assert.equal(result.stdout.trim(), version);
});

test('default uv command pins the Python package and preserves arguments', () => {
  const args = ['serve', '--model', 'directory with spaces; $(no-shell)', '--port', '9000'];
  assert.deepEqual(launchCommand(args, {}), {
    command: 'uv', args: ['tool', 'run', '--python', '3.11', '--from',
      `sokudan[serve]==${version}`, 'sokudan', ...args],
  });
  assert.match(launchCommand(['probe-position'], {}).args[5], /train,bench,torch/);
});

test('an explicit Python executable avoids PATH recursion', () => {
  assert.deepEqual(launchCommand(['serve', '--help'], { SOKUDAN_PYTHON: '/path with spaces/python' }), {
    command: '/path with spaces/python', args: ['-m', 'sokudan.cli', 'serve', '--help'],
  });
});

test('missing uv or explicit Python produces a useful nonzero failure', () => {
  const missingUv = run(['serve'], { PATH: '', SOKUDAN_PYTHON: '' });
  assert.equal(missingUv.status, 1);
  assert.match(missingUv.stderr, /Install uv/);
  const missingPython = run(['serve'], { SOKUDAN_PYTHON: join(tmpdir(), 'sokudan-nonexistent-python') });
  assert.equal(missingPython.status, 1);
  assert.match(missingPython.stderr, /could not start/);
  assert.doesNotMatch(missingPython.stderr, /Install uv/);
});

// A real Python process runs a tiny module without installing the model runtime.
function childFixture(t, source) {
  const dir = mkdtempSync(join(tmpdir(), 'sokudan cli '));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const moduleDir = join(dir, 'sokudan');
  mkdirSync(moduleDir);
  writeFileSync(join(moduleDir, '__init__.py'), '');
  writeFileSync(join(moduleDir, 'cli.py'), source);
  return { ...process.env, SOKUDAN_PYTHON: process.env.TEST_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3'), PYTHONPATH: dir };
}

test('spawned child receives literal arguments and its exit status is preserved', (t) => {
  const env = childFixture(t, 'import sys, json\nprint(json.dumps(sys.argv[1:]))\nsys.exit(7)\n');
  const args = ['serve', '--model', '日本語 directory; $(not-a-command)', '--port', '8123'];
  const result = run(args, env);
  assert.equal(result.status, 7, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), args);
});

// Windows TerminateProcess has no POSIX signal-handler semantics.
if (process.platform !== 'win32') test('termination is forwarded to the running child', async (t) => {
  const env = childFixture(t, 'import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *_: sys.exit(23))\nprint("ready", flush=True)\ntime.sleep(30)\n');
  const child = spawn(process.execPath, [cli, 'serve'], { env, stdio: ['ignore', 'pipe', 'pipe'] });
  t.after(() => child.kill('SIGKILL'));
  const closed = once(child, 'close');
  await once(child.stdout, 'data');
  child.kill('SIGTERM');
  const [code] = await closed;
  assert.equal(code, 23);
});
