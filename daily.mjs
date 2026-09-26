#!/usr/bin/env node
/**
 * daily.mjs — the zero-Claude-token daily routine in one command.
 *
 *   1. scan          node scan.mjs — new postings into data/pipeline.md
 *   2. liveness      node check-liveness.mjs --no-fallback — expired pending → Processed
 *   3. triage        node ollama-triage.mjs --cache data/triage-cache.jsonl (local model)
 *   4. apply         blacklisted → Processed (FAIL stays pending, listed for review)
 *   5. follow-ups    node followup-cadence.mjs — overdue/urgent applications
 *   6. health        node verify-pipeline.mjs
 *   → output/daily-YYYY-MM-DD.md digest, data/daily.log, macOS notification
 *
 * PASS, MARGINAL and FAIL stay pending for you. FAIL is listed as "likely no-go"
 * rather than closed: in testing ~1 in 7 local verdicts was wrong (2026-09-26),
 * so closing one is your call. `/career-ops pipeline` evaluates the
 * PASS entries (the only step that spends Claude tokens) and reuses the cached
 * triage verdicts instead of re-running the model. Nothing here evaluates,
 * writes CVs, applies, or sends anything.
 *
 * Pipeline edits use the same line formats as modes/pipeline.md and run under
 * the shared pipeline lock, so a scan or pipeline run cannot interleave.
 *
 * Usage:
 *   node daily.mjs                  # full routine
 *   node daily.mjs --skip-scan      # re-triage the current inbox only
 *   node daily.mjs --skip-liveness
 *   node daily.mjs --dry-run        # scan --dry-run, no pipeline/discard-log writes
 */

import { spawnSync } from 'child_process';
import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync, rmSync, statSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { tmpdir } from 'os';
import { getCareerOpsRoot } from './path-resolver.mjs';
import { withPipelineLock } from './pipeline-lock.mjs';
import { hasFlag, validateFlags } from './lib/cli-flags.mjs';
import { isMainModule } from './lib/is-main-module.mjs';
import { cleanField, parsePending } from './ollama-triage.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const DATA_ROOT = getCareerOpsRoot();
const PATHS = {
  pipeline: join(DATA_ROOT, 'data', 'pipeline.md'),
  triageCache: join(DATA_ROOT, 'data', 'triage-cache.jsonl'),
  scanRuns: join(DATA_ROOT, 'data', 'scan-runs.tsv'),
  log: join(DATA_ROOT, 'data', 'daily.log'),
  lock: join(DATA_ROOT, 'data', '.daily.lock'),
  outDir: join(DATA_ROOT, 'output'),
};
const LOCK_STALE_MS = 3 * 60 * 60 * 1000;

const USAGE = `
  daily.mjs — scan → liveness → local triage → follow-ups → digest (zero Claude tokens)

  node daily.mjs [--skip-scan] [--skip-liveness] [--dry-run]
`;

const today = () => new Date().toISOString().slice(0, 10);
const stamp = () => new Date().toISOString().replace('T', ' ').slice(0, 19);
const log = (m) => { appendFileSync(PATHS.log, `${m}\n`); process.stderr.write(`${m}\n`); };

/** Run a sibling script; its output goes to the log, and is returned for parsing. */
function run(script, args = [], { timeoutMs = 30 * 60_000 } = {}) {
  const t0 = Date.now();
  const r = spawnSync(process.execPath, [join(ROOT, script), ...args], {
    cwd: ROOT, encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024, timeout: timeoutMs,
  });
  appendFileSync(PATHS.log, `--- ${script} ${args.join(' ')} (exit ${r.status}, ${Math.round((Date.now() - t0) / 1000)}s)\n${r.stdout ?? ''}${r.stderr ?? ''}\n`);
  return { status: r.status, stdout: r.stdout ?? '', stderr: r.stderr ?? '', error: r.error };
}

/**
 * A scheduled run can fire right after wake/login, before the Ollama app is up.
 * Start the macOS app if the API does not answer, and wait for it (≤60 s).
 */
async function ensureOllama() {
  const base = (process.env.OLLAMA_BASE_URL || 'http://localhost:11434').replace(/\/$/, '');
  const up = async () => {
    try { return (await fetch(`${base}/api/tags`, { signal: AbortSignal.timeout(3_000) })).ok; } catch { return false; }
  };
  if (await up()) return true;
  if (!existsSync('/Applications/Ollama.app')) return false;
  log('Ollama not answering — starting the Ollama app…');
  spawnSync('open', ['-g', '-a', 'Ollama']);
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 2_000));
    if (await up()) return true;
  }
  return false;
}

const lastLine = (s) => String(s).trim().split('\n').filter(Boolean).pop() ?? '';

// ---------------------------------------------------------------------------
// Pure helpers (exported for --self-test)
// ---------------------------------------------------------------------------

/** check-liveness.mjs lines: "❌ expired    (api) https://…" → Map(url → result). */
export function parseLiveness(text) {
  const out = new Map();
  for (const line of String(text).split('\n')) {
    const m = line.match(/\b(active|expired|uncertain)\b\s+(?:\([^)]*\)\s+)?(https?:\/\/\S+)/);
    if (m) out.set(m[2], m[1]);
  }
  return out;
}

/** new_added from the last data/scan-runs.tsv row, looked up by header name. */
export function lastScanNewAdded(tsv) {
  const rows = String(tsv).trim().split('\n');
  if (rows.length < 2) return null;
  const header = rows[0].split('\t');
  const idx = header.indexOf('new_added');
  const n = Number(rows[rows.length - 1].split('\t')[idx]);
  return idx === -1 || !Number.isFinite(n) ? null : n;
}

/**
 * Move pending lines to Processed. `moves` maps URL → replacement Processed
 * line. Returns the new file text and the URLs actually moved.
 */
export function movePendingToProcessed(text, moves) {
  const lines = String(text).split('\n');
  const moved = [];
  const processedLines = [];
  for (let i = lines.length - 1; i >= 0; i--) {
    if (!lines[i].startsWith('- [ ] ')) continue;
    const url = lines[i].slice(6).split('|')[0].trim();
    if (!moves.has(url) || moved.includes(url)) continue;
    lines.splice(i, 1);
    processedLines.unshift(moves.get(url));
    moved.push(url);
  }
  let p = lines.findIndex((l) => /^##\s+(Processed|Procesadas)/i.test(l));
  if (p === -1) { lines.push('', '## Processed'); p = lines.length - 1; }
  lines.splice(p + 1, 0, ...processedLines);
  return { text: lines.join('\n'), moved };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(args) {
  validateFlags(args, ['--skip-scan', '--skip-liveness', '--dry-run', '--self-test', '--help', '-h'], USAGE);
  if (hasFlag(args, '--help') || hasFlag(args, '-h')) { console.log(USAGE); return; }
  const dryRun = hasFlag(args, '--dry-run');

  // One run at a time (a manual run and the scheduled one can overlap).
  try {
    mkdirSync(PATHS.lock);
  } catch {
    if (Date.now() - statSync(PATHS.lock).mtimeMs < LOCK_STALE_MS) {
      console.error(`daily.mjs is already running (lock ${PATHS.lock}). Remove it if that run died.`);
      process.exit(1);
    }
    rmSync(PATHS.lock, { recursive: true, force: true });
    mkdirSync(PATHS.lock);
  }

  const started = Date.now();
  const d = { date: today(), problems: [] };
  try {
    log(`\n=== ${stamp()} daily run${dryRun ? ' (dry-run)' : ''} ===`);

    // 1. Scan
    if (hasFlag(args, '--skip-scan')) {
      d.scan = 'skipped';
    } else {
      log('1/6 scan…');
      const r = run('scan.mjs', dryRun ? ['--dry-run'] : []);
      if (r.status !== 0) d.problems.push(`scan.mjs exited ${r.status}: ${lastLine(r.stderr || r.stdout)}`);
      const added = !dryRun && existsSync(PATHS.scanRuns) ? lastScanNewAdded(readFileSync(PATHS.scanRuns, 'utf-8')) : null;
      d.scan = added === null ? 'done' : `${added} new posting(s)`;
    }

    // 2. Liveness sweep (headless only: no browser window on an unattended run)
    const moves = new Map();
    const pending = parsePending(readFileSync(PATHS.pipeline, 'utf-8'));
    d.expired = [];
    if (hasFlag(args, '--skip-liveness') || pending.length === 0) {
      d.liveness = hasFlag(args, '--skip-liveness') ? 'skipped' : 'nothing pending';
    } else {
      log(`2/6 liveness sweep over ${pending.length} pending…`);
      const urlFile = join(tmpdir(), `career-ops-daily-${process.pid}.txt`);
      writeFileSync(urlFile, pending.map((e) => e.url).join('\n'));
      const r = run('check-liveness.mjs', ['--no-fallback', '--file', urlFile]);
      rmSync(urlFile, { force: true });
      const verdicts = parseLiveness(r.stdout);
      for (const e of pending) {
        if (verdicts.get(e.url) !== 'expired') continue;
        d.expired.push(e);
        moves.set(e.url, `- [x] ~~${e.url} | ${e.company} | ${e.role}~~ — posting expired (liveness sweep)`);
      }
      const uncertain = [...verdicts.values()].filter((v) => v === 'uncertain').length;
      d.liveness = `${verdicts.size} checked, ${d.expired.length} expired, ${uncertain} uncertain (left pending)`;
      if (verdicts.size === 0) d.problems.push(`liveness check produced no verdicts: ${lastLine(r.stderr || r.stdout)}`);
    }
    if (moves.size && !dryRun) {
      await withPipelineLock(PATHS.pipeline, async () => {
        writeFileSync(PATHS.pipeline, movePendingToProcessed(readFileSync(PATHS.pipeline, 'utf-8'), moves).text);
      });
    }

    // 3. Local triage (cached; only new postings reach the model)
    log('3/6 local triage…');
    d.triage = [];
    if (!(await ensureOllama())) d.problems.push('Ollama is not running and could not be started');
    const t = run('ollama-triage.mjs', ['--json', '--cache', PATHS.triageCache], { timeoutMs: 3 * 60 * 60_000 });
    if (t.status === 0 && t.stdout.trim()) {
      try { d.triage = JSON.parse(t.stdout); } catch { d.problems.push('ollama-triage.mjs printed unparseable JSON'); }
      if (dryRun) d.triage = d.triage.filter((r) => !moves.has(r.url));
    } else if (!/Nothing pending/.test(t.stderr)) {
      d.problems.push(`local triage did not run: ${lastLine(t.stderr) || `exit ${t.status}`} — /career-ops pipeline will fall back to its standard pre-screen`);
    }

    // 4. Close blacklisted postings (house rule in modes/_custom.md).
    // FAIL is not closed here: it goes to the digest's review list.
    const by = (v, pred = () => true) => d.triage.filter((r) => r.verdict === v && pred(r));
    const isBlacklisted = (r) => /^Blacklisted:/.test(r.reason);
    const closeMoves = new Map();
    for (const r of by('SKIP', isBlacklisted)) {
      closeMoves.set(r.url, `- [x] #-- | ${r.url} | skipped (blacklisted: ${cleanField(r.company, 60)} — ${cleanField(r.role, 80)})`);
    }
    d.closedByTriage = [];
    if (closeMoves.size) {
      log(`4/6 closing ${closeMoves.size} blacklisted posting(s)…`);
      if (dryRun) {
        d.closedByTriage = [...closeMoves.keys()];
      } else {
        await withPipelineLock(PATHS.pipeline, async () => {
          const res = movePendingToProcessed(readFileSync(PATHS.pipeline, 'utf-8'), closeMoves);
          writeFileSync(PATHS.pipeline, res.text);
          d.closedByTriage = res.moved;
        });
      }
    }

    // 5. Follow-ups
    log('5/6 follow-ups…');
    d.followups = [];
    const f = run('followup-cadence.mjs');
    try {
      d.followups = (JSON.parse(f.stdout).entries ?? []).filter((e) => ['overdue', 'urgent'].includes(e.urgency));
    } catch { d.problems.push(`followup-cadence.mjs: ${lastLine(f.stderr) || 'unparseable output'}`); }

    // 6. Health
    log('6/6 health check…');
    const v = run('verify-pipeline.mjs');
    d.health = (v.stdout.match(/Pipeline Health:.*$/m)?.[0] ?? `exit ${v.status}`).replace('Pipeline Health: ', '');

    // Digest
    const pass = by('PASS');
    const marginal = by('MARGINAL');
    const unreadable = d.triage.filter((r) => (r.verdict === 'SKIP' && !isBlacklisted(r)) || r.verdict === 'ERROR');
    const fails = by('FAIL');
    const blocked = by('SKIP', isBlacklisted);
    const item = (r) => `- **${r.company || 'Unknown'} — ${r.role || 'Unknown'}** · ${r.score == null ? '—' : r.score.toFixed(1)}/5 · ${cleanField(r.reason, 220)}\n  ${r.url}`;
    const section = (title, rows, fmt = item, empty = '_None._') => `## ${title} (${rows.length})\n\n${rows.length ? rows.map(fmt).join('\n') : empty}\n`;
    const md = [
      `# career-ops daily — ${d.date}${dryRun ? ' (dry-run)' : ''}`,
      '',
      `Scan: ${d.scan} · Liveness: ${d.liveness} · Health: ${d.health} · Took ${Math.round((Date.now() - started) / 60000)} min`,
      '',
      d.problems.length ? `> **Problems**\n${d.problems.map((p) => `> - ${p}`).join('\n')}\n` : '',
      section('Worth a full evaluation — PASS', pass, item,
        '_None today._'),
      pass.length ? `Next: run \`/career-ops pipeline\` to evaluate these (≈$3 of Claude usage each).\n` : '',
      section('Your call — MARGINAL', marginal),
      section('Likely no-go — FAIL (still pending; tell Claude which to close)', fails),
      section('Could not read — will get the standard pre-screen', unreadable),
      section('Follow-ups due', d.followups, (e) => `- #${e.num} **${e.company} — ${e.role}** · ${e.status} · ${e.urgency}${e.nextFollowupDate ? ` (due ${e.nextFollowupDate})` : ''}`),
      `## Closed automatically\n`,
      `- Expired (liveness): ${d.expired.length ? d.expired.map((e) => `${e.company || e.url} — ${e.role}`).join('; ') : 'none'}`,
      `- Blacklisted: ${blocked.length ? blocked.map((r) => `${r.company} — ${r.role}`).join('; ') : 'none'}`,
      '',
      `Wrong call on an auto-closed posting? Move its line from "Processed" back to "Pending" in data/pipeline.md.`,
      '',
    ].join('\n');
    mkdirSync(PATHS.outDir, { recursive: true });
    const digestPath = join(PATHS.outDir, `daily-${d.date}.md`);
    writeFileSync(digestPath, md);

    const summary = `PASS ${pass.length} · MARGINAL ${marginal.length} · FAIL ${fails.length} to review · closed ${d.expired.length + d.closedByTriage.length} · follow-ups ${d.followups.length}${d.problems.length ? ` · ⚠️ ${d.problems.length} problem(s)` : ''}`;
    log(`=== ${stamp()} done: ${summary} → ${digestPath}`);
    console.log(`${summary}\n${digestPath}`);
    spawnSync('osascript', ['-e', `display notification ${JSON.stringify(summary)} with title "career-ops daily"`]);
  } finally {
    rmSync(PATHS.lock, { recursive: true, force: true });
  }
}

function selfTest() {
  let failed = 0;
  const check = (name, ok) => { console.log(`${ok ? '✅' : '❌'} ${name}`); if (!ok) failed++; };

  const live = parseLiveness('Checking…\n✅ active     (api) https://a.test/1\n❌ expired    (api) https://a.test/2\n⚠️ uncertain  https://a.test/3\nResults: 1 active');
  check('liveness lines parsed', live.get('https://a.test/1') === 'active' && live.get('https://a.test/2') === 'expired' && live.get('https://a.test/3') === 'uncertain' && live.size === 3);

  check('scan new_added by header', lastScanNewAdded('timestamp\tstatus\tnew_added\n2026\tcompleted\t4\n2026\tcompleted\t7') === 7);
  check('scan new_added missing column → null', lastScanNewAdded('a\tb\n1\t2') === null);

  const pipe = '# P\n\n## Pending\n- [ ] https://a.test/1 | A | R1\n- [ ] https://a.test/2 | B | R2\n\n## Processed\n- [x] #001 | https://old.test | X | Y | 4.0/5 | PDF ✅';
  const res = movePendingToProcessed(pipe, new Map([['https://a.test/2', '- [x] #-- | https://a.test/2 | skipped (x)'], ['https://nope.test', 'n']]));
  check('only matching pending lines move', res.moved.length === 1 && res.moved[0] === 'https://a.test/2');
  check('moved line lands at top of Processed', /## Processed\n- \[x\] #-- \| https:\/\/a\.test\/2/.test(res.text));
  check('other pending lines untouched', res.text.includes('- [ ] https://a.test/1 | A | R1') && !res.text.includes('- [ ] https://a.test/2'));

  console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
  process.exit(failed ? 1 : 0);
}

if (isMainModule(import.meta.url)) {
  const args = process.argv.slice(2);
  if (hasFlag(args, '--self-test')) selfTest();
  else main(args).catch((err) => { console.error(`daily: ${err?.stack || err}`); process.exit(1); });
}
