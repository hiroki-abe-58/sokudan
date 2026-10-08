# sokudan for JavaScript and TypeScript

Typed HTTP client and CLI launcher for the [sokudan Japanese decision model](https://github.com/hiroki-abe-58/sokudan).
Node.js 22 or later. ESM. No npm runtime dependencies.

```sh
npm install sokudan
```

## Start a local server

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then:

```sh
npx sokudan serve --port 8000
```

The CLI runs the matching Python release (`sokudan[serve]==0.3.0`) with Python 3.11
in uv's isolated tool cache. uv can download Python if needed. The first invocation
downloads Python dependencies; starting the server also downloads the model from
Hugging Face. Later invocations reuse the caches. `npm install` itself downloads
neither Python nor model weights. `sokudan --help` and `--version` work without uv.

Apple silicon with macOS 14+ uses MLX by default. Other platforms use PyTorch.
For an existing environment (including a particular CUDA build), install
`sokudan[serve]` there and set **SOKUDAN_PYTHON** to its Python executable:

```sh
# macOS / Linux
SOKUDAN_PYTHON=/path/to/venv/bin/python npx sokudan serve
```

```powershell
# Windows PowerShell
$env:SOKUDAN_PYTHON = 'C:\path to\venv\Scripts\python.exe'
npx sokudan serve
```

This override uses that environment's sokudan version; the launcher does not
install into or modify it. Specify an executable path, not a shell command.
All server options pass through unchanged (`sokudan serve --help`). Exit codes
are preserved; on POSIX systems SIGINT/SIGTERM are forwarded to the child.
`probe-position` uses the additional Python `train,bench,torch` extras with uv;
install those extras yourself when using `SOKUDAN_PYTHON` for research probes.

## TypeScript SDK

The SDK calls an already running server. It does not start Python or perform
inference inside JavaScript. It can connect to either a local or remote server.

```ts
import { Sokudan, SokudanHTTPError } from 'sokudan';

const client = new Sokudan({ baseURL: 'http://127.0.0.1:8000' });
const result = await client.predict({
  state: '先月の請求で同じ金額が二回引き落とされています。',
  questions: {
    department: {
      type: 'choice',
      instructions: 'この問い合わせはどの部署が担当すべきか',
      criteria: { 請求: '支払い・返金', 技術: '不具合・障害', 営業: '料金・新規契約' },
    },
    urgency: {
      type: 'score', instructions: '対応の緊急度', criteria: ['通常', '早め', '緊急'],
    },
    churn: { type: 'noul', instructions: '解約を示唆しているか' },
  },
});

console.log(result.answers.department.choice); // typed as '請求' | '技術' | '営業'
console.log(result.answers.urgency.score);
console.log(result.answers.churn.noul);
console.log(await client.health());
```

- `predict(request, { signal? })` sends `POST /v1/systemone`.
- `health({ signal? })` sends `GET /health`; HTTP 503 throws just like other errors.
- Options: `baseURL` (server root, defaults to `http://127.0.0.1:8000`),
  `timeoutMs` (defaults to 120000), `apiKey`, and an optional `fetch` implementation.
  A reverse-proxy prefix such as `https://example.com/sokudan/` is preserved.
- `bool` is an input alias for `noul`; the response type is always `noul`.
- Prefer string state. Structured state is accepted but changes the model's input
  rendering. The optional request `model` is accepted but ignored by the server.
- Types follow [sokudan's wire format](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/systemone_wire_format.md).
  Validation is performed by the server; the SDK does not recompute probabilities,
  calibration or confidence, or perform runtime response-schema validation.
- Node.js is the supported runtime. Browser use additionally requires CORS support
  from your server/reverse proxy; the default server does not enable it.

```ts
try {
  await client.predict({
    state: '確認したい文章',
    questions: { ok: { type: 'bool', instructions: '依頼が含まれているか' } },
  }, { signal: AbortSignal.timeout(5000) });
} catch (error) {
  if (error instanceof SokudanHTTPError) {
    console.error(error.status, error.body, error.headers.get('retry-after'));
  } else {
    throw error; // network, cancellation, timeout or malformed JSON
  }
}
```

There are no automatic retries. HTTP errors retain the response body and headers.
The request deadline includes reading the body. `apiKey` sends a bearer header for
an authenticating reverse proxy; **sokudan itself does not validate this header**.

## Develop and publish

From `packages/npm`: `npm ci`, `npm run typecheck`, `npm test`, `npm pack`.
Tests use Node's test runner and a Python 3 executable (no model dependencies).
`TEST_PYTHON` can select that interpreter. `npm pack` includes compiled JavaScript,
TypeScript declarations, README, LICENSE and NOTICE.

See [distribution and release instructions](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/distribution.md).
