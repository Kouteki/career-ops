/**
 * ollama-client.mjs — shared Ollama endpoint resolution with server->local failover.
 *
 * Precedence (matches the CAREER_OPS_ROOT-style "explicit env var always wins"
 * convention used elsewhere in this repo — see AGENTS.md, Path Resolution Override):
 *
 *   1. OLLAMA_BASE_URL   — explicit override. Used as-is, no probe, no failover.
 *                          A remote host here still requires OLLAMA_ALLOW_REMOTE=1
 *                          (unchanged pre-existing safety behavior).
 *   2. OLLAMA_SERVER_URL — optional home/office server running Ollama. Probed
 *                          first (reachable + model pulled, when a model is
 *                          given); on any failure, falls back to local. Setting
 *                          this var IS the deliberate opt-in to a remote
 *                          endpoint, so it does not require OLLAMA_ALLOW_REMOTE.
 *   3. Local default: http://localhost:11434
 *
 * Callers that already send a caller-specific payload (a triage brief, a JD)
 * to the resolved endpoint should still treat "server" vs "local" as data-sent-
 * off-this-machine information worth logging, since OLLAMA_SERVER_URL is
 * typically a LAN/VPN address, not literally loopback.
 */

export const LOCAL_OLLAMA_URL = 'http://localhost:11434';

const LOOPBACK_HOSTS = ['localhost', 'localhost.localdomain', '127.0.0.1', '::1', '[::1]'];

function stripTrailingSlash(url) {
  return url.replace(/\/$/, '');
}

export function isLoopbackHost(hostname) {
  return LOOPBACK_HOSTS.includes(String(hostname ?? '').toLowerCase());
}

/** Throws when `baseUrl` is non-local and OLLAMA_ALLOW_REMOTE is not set. */
export function assertLoopbackOrAllowed(baseUrl) {
  const host = (() => { try { return new URL(baseUrl).hostname; } catch { return ''; } })();
  if (!isLoopbackHost(host) && process.env.OLLAMA_ALLOW_REMOTE !== '1') {
    throw new Error(`${baseUrl} is not a local endpoint; data sent to it would leave this machine. Set OLLAMA_ALLOW_REMOTE=1 if intended.`);
  }
}

/**
 * Reachability + (optional) model-pulled probe against one candidate endpoint.
 * Never throws — returns { ok, reason }.
 */
export async function probeOllama(baseUrl, { timeoutMs = 5_000, model } = {}) {
  try {
    const res = await fetch(`${stripTrailingSlash(baseUrl)}/api/tags`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { ok: false, reason: `HTTP ${res.status}` };
    if (model) {
      const body = await res.json();
      const names = (body.models || []).map((m) => m.name);
      if (!names.includes(model) && !names.includes(`${model}:latest`)) {
        return { ok: false, reason: `model "${model}" not pulled` };
      }
    }
    return { ok: true, reason: null };
  } catch (err) {
    return { ok: false, reason: err.name === 'TimeoutError' ? 'timeout' : err.message };
  }
}

/**
 * Resolve which Ollama base URL (and model) to use for this run.
 *
 * A server and a local machine commonly run different-sized models (a
 * dedicated server can host a much bigger one than a laptop), so the model
 * to request is resolved together with the endpoint, not assumed constant.
 *
 * @param {object} opts
 * @param {number} [opts.timeoutMs]    probe timeout per candidate
 * @param {string} [opts.model]        model to require on the LOCAL candidate (and on an explicit OLLAMA_BASE_URL)
 * @param {string} [opts.serverModel]  model to require on the SERVER candidate (OLLAMA_SERVER_URL); defaults to `model` when omitted, so behavior is unchanged unless the caller (or OLLAMA_SERVER_MODEL) sets a different one
 * @param {(msg: string) => void} [opts.log]  called with one-line status notes (e.g. "falling back to local")
 * @returns {Promise<{ baseUrl: string, source: 'explicit'|'server'|'local', model: string|undefined, attempts: Array<{baseUrl:string, ok:boolean, reason:string|null}> }>}
 * @throws when an explicit OLLAMA_BASE_URL is remote and disallowed, or when
 *         every candidate (server, then local) is unreachable.
 */
export async function resolveOllamaBaseUrl({ timeoutMs = 5_000, model, serverModel, log = () => {} } = {}) {
  const explicit = process.env.OLLAMA_BASE_URL?.trim();
  if (explicit) {
    const baseUrl = stripTrailingSlash(explicit);
    assertLoopbackOrAllowed(baseUrl);
    return { baseUrl, source: 'explicit', model, attempts: [] };
  }

  const attempts = [];
  const server = process.env.OLLAMA_SERVER_URL?.trim();
  if (server) {
    const serverUrl = stripTrailingSlash(server);
    const modelForServer = serverModel || process.env.OLLAMA_SERVER_MODEL?.trim() || model;
    const probe = await probeOllama(serverUrl, { timeoutMs, model: modelForServer });
    attempts.push({ baseUrl: serverUrl, ...probe });
    if (probe.ok) return { baseUrl: serverUrl, source: 'server', model: modelForServer, attempts };
    log(`Server Ollama unreachable (${serverUrl}: ${probe.reason}) — falling back to local`);
  }

  const localUrl = LOCAL_OLLAMA_URL;
  const probe = await probeOllama(localUrl, { timeoutMs, model });
  attempts.push({ baseUrl: localUrl, ...probe });
  if (probe.ok) return { baseUrl: localUrl, source: 'local', model, attempts };

  const detail = attempts.map((a) => `${a.baseUrl} (${a.reason})`).join('; ');
  throw new Error(`No Ollama endpoint reachable: ${detail}`);
}
