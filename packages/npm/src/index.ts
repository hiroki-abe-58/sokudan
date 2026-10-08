import type { HealthResponse, Questions, SystemOneRequest, SystemOneResponse } from "./types.js";

export type * from "./types.js";

export interface SokudanOptions {
  /** Server root, optionally including a reverse-proxy path prefix. */
  baseURL?: string;
  /** For an authenticating reverse proxy; sokudan itself does not check this key. */
  apiKey?: string;
  /** Request deadline, including reading the body. Default: 120 seconds. */
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

export class SokudanHTTPError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
    public readonly headers: Headers,
  ) {
    super(`sokudan HTTP ${status}: ${typeof body === "string" ? body : JSON.stringify(body)}`);
    this.name = "SokudanHTTPError";
  }
}

/** HTTP client for `sokudan serve`; inference runs on that server. */
export class Sokudan {
  private readonly baseURL: URL;
  private readonly headers: Headers;
  private readonly timeoutMs: number;
  private readonly fetcher: typeof globalThis.fetch;

  constructor(options: SokudanOptions = {}) {
    this.baseURL = new URL(options.baseURL ?? "http://127.0.0.1:8000");
    if (!['http:', 'https:'].includes(this.baseURL.protocol)
        || this.baseURL.search || this.baseURL.hash) {
      throw new TypeError("baseURL must be an HTTP(S) server root without a query or fragment");
    }
    if (!this.baseURL.pathname.endsWith("/")) this.baseURL.pathname += "/";
    this.timeoutMs = options.timeoutMs ?? 120_000;
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs <= 0 || this.timeoutMs > 2_147_483_647) {
      throw new TypeError("timeoutMs must be an integer between 1 and 2147483647");
    }
    this.headers = new Headers({ Accept: "application/json" });
    if (options.apiKey) this.headers.set("Authorization", `Bearer ${options.apiKey}`);
    this.fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  predict<const Q extends Questions>(
    request: SystemOneRequest<Q>, options: RequestOptions = {},
  ): Promise<SystemOneResponse<Q>> {
    return this.request("v1/systemone", request, options);
  }

  health(options: RequestOptions = {}): Promise<HealthResponse> {
    return this.request("health", undefined, options);
  }

  private async request<T>(path: string, body: unknown, options: RequestOptions): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new DOMException(
      "sokudan request timed out", "TimeoutError",
    )), this.timeoutMs);
    const signal = options.signal
      ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
    const headers = new Headers(this.headers);
    if (body !== undefined) headers.set("Content-Type", "application/json");
    try {
      const response = await this.fetcher(new URL(path, this.baseURL), {
        method: body === undefined ? "GET" : "POST",
        headers,
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        signal,
      });
      const text = await response.text();
      let parsed: unknown;
      try { parsed = JSON.parse(text); }
      catch (error) {
        if (response.ok) throw error;
        parsed = text;
      }
      if (!response.ok) throw new SokudanHTTPError(response.status, parsed, response.headers);
      // The server owns validation. Keep its scores, calibration and extra fields intact.
      return parsed as T;
    } finally {
      clearTimeout(timer);
    }
  }
}
