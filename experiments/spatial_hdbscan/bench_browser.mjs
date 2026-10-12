// Runtime of the auto-label methods in the real TEE viewer, in Chrome (Christopher, 12 Oct 2026).
// Starts a separate headless Chrome with an empty profile, opens the local TEE viewer on the
// Gothenburg viewport, switches to the labelling layout and presses "Go" programmatically
// (window.runKMeans) for every method. One timed run = from the call until TEE has drawn the
// cluster overlay and filled the cluster list, i.e. what a user waits for after pressing "Go":
// de-quantising the embeddings, drawing the samples, the Web Worker, and the overlay.
//
//   node bench_browser.mjs [reps=9] [viewport=Gteborg]
//
// Needs the local TEE at http://127.0.0.1:8001 (TESSERA/Local/local-tee start).
// A headless page counts as visible; a hidden tab in a normal Chrome window is throttled and
// gives 9-15 s per run, which says nothing about the methods.
// Chrome's CPU throttling (DevTools "4x slowdown") is no substitute for slower hardware here: it
// slows only the main thread, not the Web Worker (tried on 12 Oct: every method +0.25 s).
// Writes results/bench_browser_<viewport>.json.
import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const reps = +(process.argv[2] || 9);
const viewport = process.argv[3] || 'Gteborg';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9333 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'tee-bench-chrome-'));

const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, '--no-first-run',
    '--no-default-browser-check', '--window-size=1600,1000', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cleanup = () => { try { chrome.kill(); } catch { /* already gone */ } try { fs.rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ } };
process.on('exit', cleanup);

let target;
for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === 'page'); } catch { /* not up yet */ }
}
if (!target) { console.error('Chrome did not start'); process.exit(1); }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => { ws.onopen = r; });
let id = 0;
const pending = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
    const m = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (m.error || m.result.exceptionDetails) throw new Error(JSON.stringify(m.error || m.result.exceptionDetails));
    return m.result.result.value;
};

await send('Page.enable');
await send('Page.navigate', { url: `http://127.0.0.1:8001/viewer.html?viewport=${viewport}` });
let ready = false;
for (let i = 0; i < 120 && !ready; i++) { await sleep(500); ready = await evaluate('!!(window.localVectors && window.runKMeans && window.SEG_METHODS)').catch(() => false); }
if (!ready) { console.error('viewer did not load the vectors'); process.exit(1); }
await evaluate(`(() => { const s = document.getElementById('panel-layout-select'); s.value = 'labelling'; s.dispatchEvent(new Event('change', { bubbles: true })); })()`);
await sleep(1500);
const info = await evaluate(`({ N: window.localVectors.numVectors, dim: window.localVectors.dim, visibility: document.visibilityState, mode: window.currentPanelMode,
    chrome: navigator.userAgent.replace(/^.*(Chrome\\/[\\d.]+).*$/, '$1'), cores: navigator.hardwareConcurrency, methods: Object.keys(window.SEG_METHODS) })`);
console.log(info);

// One run: resolved by a MutationObserver on the "Clear" button, which TEE shows as the last step
// of a finished run. No timers involved.
const runOnce = (k, method, seed) => evaluate(`new Promise((resolve) => {
    const btn = document.getElementById('seg-clear-btn');
    btn.style.display = 'none';
    const obs = new MutationObserver(() => { if (btn.style.display === 'none') return; obs.disconnect(); resolve(performance.now() - t0); });
    obs.observe(btn, { attributes: true, attributeFilter: ['style'] });
    const t0 = performance.now();
    window.runKMeans(${k}, ${seed}, '${method}');
})`);
const shares = () => evaluate(`window.segLabels.map((c) => +(100 * c.count / window.segVectors.numVectors).toFixed(1)).sort((a, b) => b - a)`);

const stats = (ts) => { const s = ts.slice().sort((a, b) => a - b), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))]; return { median: q(0.5), p25: q(0.25), p75: q(0.75), min: s[0], max: s[s.length - 1] }; };
const res = { viewport, ...info, reps, cpu: os.cpus()[0].model, date: new Date().toISOString().slice(0, 10), k: {}, shares_seed42_k5: {} };
for (const m of info.methods) await runOnce(5, m, 1);                       // warm-up
for (const k of [5, 17]) {
    res.k[k] = {};
    for (const method of info.methods) {
        const ts = [];
        for (let r = 0; r < reps; r++) {
            ts.push(await runOnce(k, method, 42 + r));
            if (k === 5 && r === 0) res.shares_seed42_k5[method] = await shares();
        }
        res.k[k][method] = { ms: ts.map((v) => +v.toFixed(0)), ...stats(ts) };
        const s = res.k[k][method];
        console.log(`k=${String(k).padEnd(2)} ${method.padEnd(7)} median ${s.median.toFixed(0).padStart(5)} ms  [${s.p25.toFixed(0)}-${s.p75.toFixed(0)}]  min ${s.min.toFixed(0)} max ${s.max.toFixed(0)}`);
    }
}
console.log('cluster shares, seed 42, k = 5:', res.shares_seed42_k5);
fs.writeFileSync(path.join(HERE, `results/bench_browser_${viewport}.json`), JSON.stringify(res, null, 1));
ws.close();
process.exit(0);
