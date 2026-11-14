'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const fmt = (v, d = 2) => (v == null || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
const pct = (v, d = 0) => v == null ? '—' : (v * 100).toFixed(d) + '%';
const yes = v => v === true || v === 'True' || v === 'true';
const COL = ['#5d5f60', '#1b6c80', '#2f70bb', '#a86f0c'];

function el(tag, text, cls, attrs) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (cls) node.className = cls;
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}
function svg(tag, attrs, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
const card = (...kids) => { const c = el('div', null, 'card'); c.append(...kids); return c; };
const signed = (v, d = 0) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d) + '%';
const short = c => c.replace('words ', '').replace('translate to French and back (per sentence)', 'translated').replace('translate to French and back', 'translated')
  .replace('paraphrase each sentence', 'paraphrased, per sentence').replace('paraphrase whole text', 'paraphrased, whole').replace('none', 'no change');

let token = '', mode = 'real', view = 'results', last = null, polling = null, jobId = null, archive = null;

async function api(url, data) {
  const response = await fetch(url, data ? { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Dashboard-Token': token }, body: JSON.stringify(data) } : {});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

const TABS = { generate: ['Generate & compare', 'Generate & compare'], detect: ['Check text', 'Check text'], attack: ['Rewrite test', 'Run rewrite test'] };
const TITLES = { results: 'Compare', runs: 'Runs' };

function setView(next) {
  view = next;
  document.body.classList.remove('nav-open');
  for (const b of $('nav').children) b.classList.toggle('on', b.dataset.view === next);
  const bench = next in TABS;
  $('results').hidden = next !== 'results'; $('bench').hidden = !bench; $('runs').hidden = next !== 'runs';
  $('crumb').textContent = bench ? TABS[next][0] : TITLES[next];
  if (bench) {
    $('title').textContent = TABS[next][0]; $('run').textContent = TABS[next][1];
    $('in-generate').hidden = next !== 'generate'; $('in-text').hidden = next === 'generate';
    $('attack-pick').hidden = next !== 'attack'; $('use-last').hidden = !last || next === 'generate';
    $('result').replaceChildren();
  }
  if (next === 'runs') loadRuns();
  if (next === 'results') loadBenchmark();
}

function setMode(next) {
  mode = next;
  for (const b of document.querySelectorAll('.seg button')) b.classList.toggle('on', b.dataset.mode === next);
}

function settings(action) {
  const num = id => Number($(id).value);
  return {
    action, mode, scheme: $('scheme').value, model: $('model').value.trim(), revision: $('revision').value.trim() || 'main', key: $('key').value,
    prompt: $('prompt').value, text: $('text').value, attack: $('attack').value,
    gamma: num('gamma'), delta: num('delta'), threshold: num('threshold'), temperature: num('temperature'),
    top_p: num('top_p'), max_new_tokens: num('max_new_tokens'), seed: num('seed'),
    limit: action === 'benchmark' ? num('bench-limit') : (num('limit') || 1),
  };
}

/* ---------- jobs ---------- */
function chip(state, text) { const c = $('job-chip'); c.className = 'chip ' + state; c.lastElementChild.textContent = text; }
function setBusy(busy) { for (const id of ['run', 'run-exp', 'run-bench']) $(id).disabled = busy; $('cancel').hidden = !busy; }
const JOB_NAMES = { generate: 'Generating', detect: 'Scoring', attack: 'Rewriting', experiment: 'Experiment', benchmark: 'Benchmark', smoke: 'Demo' };
const outBox = () => view in TABS ? $('result') : view === 'results' ? $('bench-out') : $('run-detail');

async function start(action) {
  $('result').replaceChildren(); setBusy(true); chip('run', 'Running');
  $('job').hidden = false; $('log').textContent = ''; $('job-title').textContent = JOB_NAMES[action] || 'Working';
  const started = Date.now();
  try {
    const { id } = await api('/api/jobs', settings(action)); jobId = id;
    polling = setInterval(async () => {
      try {
        const job = await api('/api/jobs/' + id);
        $('log').textContent = job.log.trim().split('\n').slice(-5).join('\n'); $('log').scrollTop = 1e9;
        const s = Math.round((Date.now() - started) / 1000);
        $('job-time').textContent = s >= 60 ? Math.floor(s / 60) + 'm ' + (s % 60) + 's' : s + 's';
        if (job.status === 'running' || job.status === 'cancelling') return;
        clearInterval(polling); finish(job, action);
      } catch (e) { clearInterval(polling); fail(e.message); }
    }, 1000);
  } catch (e) { fail(e.message); }
}
function fail(message) { setBusy(false); chip('bad', 'Failed'); $('job').hidden = true; outBox().replaceChildren(el('div', message, 'err')); }
function finish(job, action) {
  setBusy(false); $('job').hidden = true;
  if (job.status === 'cancelled') { chip('', 'Cancelled'); return; }
  if (job.status === 'failed') {
    chip('bad', 'Failed'); const box = el('div', job.log.trim().split('\n').slice(-5).join('\n'), 'err'); box.style.whiteSpace = 'pre-wrap'; outBox().replaceChildren(box); return;
  }
  chip('ok', 'Done');
  if (action === 'benchmark') return setView('results');
  if (action === 'experiment' || action === 'smoke') return setView('runs');
  if (job.result) renderResult(job.result, action);
}

/* ---------- charts ---------- */
function hbars(rows, opts = {}) {
  const W = 720, L = opts.left || 190, R = 130, rowH = 28, H = rows.length * rowH + 8;
  const max = opts.max || Math.max(...rows.map(r => r.value), 1e-9);
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': opts.label || 'Bar chart' });
  rows.forEach((r, i) => {
    const y = 4 + i * rowH, w = Math.max(2, r.value / max * (W - L - R));
    s.append(svg('text', { x: L - 10, y: y + 17, 'text-anchor': 'end' }, r.label), svg('rect', { x: L, y: y + 4, width: w, height: 16, rx: 3, fill: r.color }), svg('text', { x: L + w + 8, y: y + 17, class: 'num' }, r.text));
  });
  return s;
}

/* One dot per text: grey = unwatermarked, colour = watermarked. Right of the dashed line = detected. */
function strip(sm, cond) {
  const names = Object.keys(sm.schemes), W = 720, L = 190, R = 84, rowH = 50, top = 26, T = sm.threshold;
  const all = names.flatMap(n => [...sm.schemes[n].conditions[cond].z, ...sm.schemes[n].control_z]);
  const hi = Math.min(30, Math.max(T * 2, ...all)), lo = Math.max(-6, Math.min(-2, ...all));
  const x = z => L + (Math.max(lo, Math.min(hi, z)) - lo) / (hi - lo) * (W - L - R);
  const H = top + names.length * rowH + 22;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': 'Score of every text after: ' + short(cond) });
  for (const t of [0, 5, 10, 20, 30].filter(t => t > lo && t < hi)) s.append(svg('line', { x1: x(t), x2: x(t), y1: top - 6, y2: H - 18, stroke: '#f0f1f2' }), svg('text', { x: x(t), y: H - 4, 'text-anchor': 'middle' }, String(t)));
  s.append(svg('line', { x1: x(T), x2: x(T), y1: top - 12, y2: H - 18, stroke: '#131416', 'stroke-dasharray': '5 4', opacity: .7 }), svg('text', { x: x(T) + 6, y: top - 8 }, 'detected →'));
  names.forEach((n, i) => {
    const cy = top + i * rowH + rowH / 2, d = sm.schemes[n], c = d.conditions[cond];
    if (i) s.append(svg('line', { x1: 0, x2: W, y1: cy - rowH / 2, y2: cy - rowH / 2, stroke: '#f0f1f2' }));
    s.append(svg('text', { x: L - 12, y: cy + 4, 'text-anchor': 'end' }, n));
    d.control_z.forEach((z, j) => { const dot = svg('circle', { cx: x(z), cy: cy - 10 + ((j * 7) % 5 - 2) * 2.4, r: 3.4, fill: '#cdced1', opacity: .9 }); dot.append(svg('title', {}, 'unwatermarked text · z ' + fmt(z, 1))); s.append(dot); });
    c.z.forEach((z, j) => { const dot = svg('circle', { cx: x(z), cy: cy + 11 + ((j * 7) % 5 - 2) * 2.4, r: 3.6, fill: COL[i % 4], opacity: .8 }); dot.append(svg('title', {}, 'watermarked text · z ' + fmt(z, 1))); s.append(dot); });
    s.append(svg('text', { x: W - R + 12, y: cy - 2, class: 'num' }, pct(c.tpr) + ' found'), svg('text', { x: W - R + 12, y: cy + 14 }, pct(d.fpr) + ' false'));
  });
  return s;
}

/* ---------- compare ---------- */
function noise(p1, p2, n) { const c = p => Math.min(.95, Math.max(.05, p)); return 2 * Math.sqrt(c(p1) * (1 - c(p1)) / n + c(p2) * (1 - c(p2)) / n); }

function verdictChips(sm) {
  const names = Object.keys(sm.schemes), S = sm.schemes, base = names[0], n = sm.samples, conds = sm.conditions.filter(c => c !== 'none');
  const row = el('div', null, 'vrow'), add = (label, big, sub, cls) => { const d = el('div', null, 'vchip ' + (cls || '')); d.append(el('span', label, 'l'), el('b', big), el('span', sub, 's')); row.append(d); };
  const fast = names.slice().sort((a, b) => S[a].overhead_pct - S[b].overhead_pct)[0];
  add('Fastest', fast, signed(S[fast].overhead_pct) + ' time vs none');
  const avg = k => conds.reduce((a, c) => a + S[k].conditions[c].tpr, 0) / conds.length;
  const best = names.slice().sort((a, b) => avg(b) - avg(a))[0];
  const clear = best !== base && avg(best) - avg(base) > noise(avg(best), avg(base), n);
  add('Survives changes best', clear ? best : 'no clear winner', clear ? pct(avg(best)) + ' avg found vs ' + pct(avg(base)) + ' for ' + base : '±' + fmt(100 * noise(.5, .5, n), 0) + ' pts noise at ' + n + ' texts', clear ? 'good' : 'flat');
  const pp = names.map(k => S[k].perplexity_vs_control_pct);
  add('Text quality', signed(Math.min(...pp)) + ' to ' + signed(Math.max(...pp)), 'perplexity vs unwatermarked');
  const fa = Math.max(...names.map(k => S[k].fpr));
  add('False alarms', pct(fa), 'worst scheme, unwatermarked text', fa > .05 ? 'bad' : '');
  return row;
}

function heat(v) { const sp = el('span', pct(v), 'hm'); sp.style.background = `rgba(27,108,128,${.06 + v * .6})`; sp.style.color = v > .65 ? '#fff' : 'inherit'; return sp; }
function th(text, tip, right) { const h = el('th', text, right ? 'n' : ''); if (tip) h.title = tip; return h; }

function scoreboard(sm) {
  const names = Object.keys(sm.schemes), S = sm.schemes, wrap = el('div', null, 'tablewrap tall'), t = el('table'), hr = el('tr');
  hr.append(th('Scheme'), th('Speed', 'Tokens per second while generating, and extra time versus no watermark', 1),
    th('Perplexity', 'How surprising the text is to the base model. Change versus unwatermarked text; lower is closer to normal', 1),
    th('False alarms', 'Share of unwatermarked texts wrongly flagged', 1));
  for (const c of sm.conditions) hr.append(th(short(c), 'Share of watermarked texts still detected after: ' + short(c), 1));
  t.append(hr);
  const row = (name, d, ctl, i) => {
    const tr = el('tr'), nm = el('td'); if (!ctl) { const sw = el('i', null, 'sw'); sw.style.background = COL[i % 4]; nm.append(sw); } nm.append(document.createTextNode(name)); tr.append(nm);
    tr.append(el('td', fmt(d.tokens_per_second, 1) + ' tok/s' + (ctl ? '' : '  ' + signed(d.overhead_pct)), 'n'),
      el('td', fmt(d.perplexity, 1) + (ctl ? '' : '  ' + signed(d.perplexity_vs_control_pct)), 'n'), el('td', ctl ? '—' : pct(d.fpr), 'n'));
    for (const c of sm.conditions) { const td = el('td', null, 'n'); if (ctl) td.textContent = '—'; else td.append(heat(d.conditions[c].tpr)); tr.append(td); }
    t.append(tr);
  };
  row('no watermark', sm.control, true);
  names.forEach((n, i) => row(n, S[n], false, i));
  wrap.append(t);
  return wrap;
}

function renderBenchmark(sm, box) {
  box.replaceChildren();
  const names = Object.keys(sm.schemes), S = sm.schemes;
  box.append(el('p', `${sm.model} · ${sm.samples} texts per scheme · ${sm.max_new_tokens} tokens · strength ${sm.delta} · ${sm.created_at.replace('T', ' ')}`, 'meta'));
  if (sm.samples < 20) { const b = el('div', `Preview: ${sm.samples} texts per scheme. Too few to compare schemes.`, 'banner'); box.append(b); }
  box.append(verdictChips(sm), card(scoreboard(sm)));

  const area = el('div'), tabs = el('div', null, 'atabs'), head = el('div', null, 'row-between');
  const legend = el('div', null, 'legend-row'); legend.innerHTML = '<span><i style="background:#cdced1"></i>unwatermarked</span><span><i style="background:#1b6c80"></i>watermarked</span><span>one dot per text · right of the dashed line = detected</span>';
  const draw = c => { area.replaceChildren(strip(sm, c)); for (const b of tabs.children) b.classList.toggle('on', b.dataset.c === c); };
  for (const c of sm.conditions) { const b = el('button', short(c), null, { 'data-c': c }); b.onclick = () => draw(c); tabs.append(b); }
  head.append(el('h2', 'Every text, after the change'), tabs);
  box.append(card(head, legend, area));
  draw(sm.conditions.find(c => c.startsWith('translate')) || sm.conditions[sm.conditions.length - 1]);

  if (sm.fidelity) {
    const rows = Object.entries(sm.fidelity).map(([c, v]) => ({ label: short(c), value: Math.max(v, 0), text: fmt(v, 2), color: '#2f70bb' }));
    box.append(card(el('h2', 'Rewrite fidelity'), el('p', '1.00 = same meaning · a different answer to the same prompt ≈ 0.67', 'meta'), hbars(rows, { max: 1, left: 200, label: 'Rewrite fidelity' })));
  }

  const opts = [['no watermark', sm.control.sample]];
  for (const n of names) { opts.push([n, S[n].sample]); if (S[n].back_translation_sample) opts.push([n + ' → rewritten', S[n].back_translation_sample]); }
  const sel = el('select'); for (const [l] of opts) sel.append(el('option', l));
  const body = el('div', opts[0][1], 'hl'); sel.onchange = () => { body.textContent = opts[sel.selectedIndex][1]; };
  const rd = el('div', null, 'row-between'); rd.append(el('h2', 'Read the text'), sel);
  box.append(card(rd, body));

  const lim = el('details', null, 'card settings'), ul = el('ul', null, 'caveats');
  for (const x of ['One model, one strength, one run.', 'Unwatermarked text here is model-written, not human.', 'Edits are random, not a person or a paraphraser.', 'No attacker adapted to the scheme.']) ul.append(el('li', x));
  lim.append(el('summary', 'Limits'), ul); box.append(lim);
}

async function loadBenchmark() {
  const box = $('bench-out'); box.replaceChildren(el('p', 'Loading…', 'meta'));
  try {
    const runs = await api('/api/runs');
    const found = runs.filter(r => r.summary && r.summary.kind === 'benchmark').sort((a, b) => (b.summary.samples - a.summary.samples) || (a.name < b.name ? 1 : -1));
    const running = runs.find(r => r.name.startsWith('benchmark-') && r.status && r.status.status === 'running' && r.status.total);
    if (!found.length) {
      const e = el('div', null, 'empty'); e.append(el('h2', running ? `Running · ${running.status.completed}/${running.status.total}` : 'No results yet'), el('p', running ? (running.status.stage || '') : 'Run benchmark, top right.'));
      box.replaceChildren(e); return;
    }
    const sel = el('select'); sel.style.cssText = 'max-width:420px;margin-bottom:12px';
    for (const r of found) sel.append(el('option', `${r.name.replace('benchmark-', '')} · ${r.summary.samples} texts · ${Object.keys(r.summary.schemes).length} schemes`));
    const body = el('div'); sel.onchange = () => renderBenchmark(found[sel.selectedIndex].summary, body);
    box.replaceChildren(...(found.length > 1 ? [sel, body] : [body])); renderBenchmark(found[0].summary, body);
  } catch (e) { box.replaceChildren(el('div', e.message, 'err')); }
}

/* ---------- test screens ---------- */
function trajectory(samples, threshold) {
  const names = Object.keys(samples), W = 720, H = 230, L = 44, R = 24, T = 30, B = 36;
  const zs = names.map(n => samples[n].stats.z_score), real = zs.filter(z => z != null);
  const top = Math.max(threshold * 1.6, ...real, 2), bottom = Math.min(-1, ...real);
  const x = i => L + (names.length === 1 ? (W - L - R) / 2 : i * (W - L - R) / (names.length - 1));
  const y = z => T + (top - z) * (H - T - B) / (top - bottom);
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': 'Score at each step' });
  for (const t of [0, Math.round(top / 2), Math.round(top)]) s.append(svg('line', { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: '#f0f1f2' }), svg('text', { x: L - 8, y: y(t) + 4, 'text-anchor': 'end' }, String(t)));
  s.append(svg('line', { x1: L, x2: W - R, y1: y(threshold), y2: y(threshold), stroke: '#131416', 'stroke-dasharray': '5 4', opacity: .6 }), svg('text', { x: W - R, y: y(threshold) - 6, 'text-anchor': 'end' }, 'detected above'));
  const pts = names.map((n, i) => [x(i), zs[i] == null ? null : y(zs[i])]).filter(p => p[1] != null);
  if (pts.length > 1) s.append(svg('polyline', { points: pts.map(p => p.join(',')).join(' '), fill: 'none', stroke: '#1b6c80', 'stroke-width': 2 }));
  names.forEach((n, i) => {
    s.append(svg('text', { x: x(i), y: H - 12, 'text-anchor': names.length > 1 && i === names.length - 1 ? 'end' : names.length > 1 && i === 0 ? 'start' : 'middle' }, n));
    if (zs[i] == null) return; const hit = samples[n].stats.detected;
    s.append(svg('circle', { cx: x(i), cy: y(zs[i]), r: 6, fill: hit ? '#1e7b4f' : '#fff', stroke: hit ? '#1e7b4f' : '#c4242a', 'stroke-width': 2 }), svg('text', { x: x(i), y: y(zs[i]) - 12, 'text-anchor': 'middle', class: 'num' }, fmt(zs[i], 1)));
  });
  return s;
}

function sampleCard(name, sample, result, threshold) {
  const st = sample.stats, c = el('div', null, 'card sample'), gamma = Number($('gamma').value);
  const verdict = st.z_score == null ? ['Too short', 'neu'] : st.detected ? ['Detected', 'pos'] : ['Not detected', 'neg'];
  const head = el('header'), left = el('div', null, 'who');
  left.append(el('h3', name), el('span', verdict[0], 'pill ' + verdict[1]));
  const big = el('div', null, 'bigz', { title: 'z-score: how far above chance the share of favored words is. Detected at ' + threshold + ' or more.' }); big.append(el('b', fmt(st.z_score, 1)), el('span', 'z'));
  head.append(left, big); c.append(head);
  const gauge = el('div', null, 'gauge'), max = Math.max(threshold * 2, st.z_score || 0, 1), fill = el('i'), mark = el('u');
  fill.style.width = Math.max(0, Math.min(100, (st.z_score || 0) / max * 100)) + '%'; mark.style.left = threshold / max * 100 + '%'; gauge.append(fill, mark); c.append(gauge);
  const body = el('div', null, 'hl'); for (const t of sample.tokens) body.append(el('span', t.text, t.status)); c.append(body);
  c.append(el('p', st.z_score == null ? 'not enough text' : `${st.green_tokens} of ${st.scored_tokens} favored · ${pct(gamma)} expected by chance`, 'meta foot'));
  const row = el('div', null, 'actions'); row.style.margin = '8px 0 0';
  if (!result.offline) { const b = el('button', 'Rewrite this', 'ghost'); b.onclick = () => useSample(sample, 'attack'); row.append(b); }
  const k = el('button', 'Check this', 'ghost'); k.onclick = () => useSample(sample, 'detect'); row.append(k); c.append(row);
  return c;
}

function renderResult(result, action) {
  const out = $('result'), threshold = Number($('threshold').value), samples = result.samples, names = Object.keys(samples);
  out.replaceChildren();
  if (action === 'generate' && samples.Watermarked) { last = samples.Watermarked; $('use-last').hidden = false; }
  if (names.length > 2) out.append(card(trajectory(samples, threshold)));
  const lg = el('div', null, 'legend-row'); lg.innerHTML = '<span><i style="background:var(--green-bg);outline:1px solid #9fd8b4"></i>favored word</span><span><i style="background:#fff;outline:1px solid #cdced1"></i>other</span><span><i style="background:#fff;outline:1px solid #cdced1;opacity:.45"></i>not counted</span>' + (result.offline ? '<span>demo: numbers are token IDs</span>' : '');
  out.append(lg);
  if (result.french) out.append(card(el('h3', 'In French'), el('p', result.french, 'small')));
  const grid = el('div', null, 'cmpgrid'); for (const n of names) grid.append(sampleCard(n, samples[n], result, threshold)); out.append(grid);
}
function useSample(sample, tab) { $('text').value = sample.text; setView(tab); window.scrollTo(0, 0); }

/* ---------- runs ---------- */
async function loadRuns() {
  const list = $('run-list'); list.replaceChildren(el('p', 'Loading…', 'meta'));
  try {
    const runs = await api('/api/runs'); list.replaceChildren();
    if (!runs.length) list.append(el('p', 'No runs.', 'meta'));
    for (const r of runs) {
      const b = el('button', null, 'run'), s = r.status || {};
      const kind = r.name.startsWith('benchmark') ? 'Benchmark' : r.offline ? 'Demo' : r.name.startsWith('research') ? 'Research' : 'Experiment';
      b.append(el('b', r.name), el('span', kind + ' · ' + (s.status || 'saved') + (s.scored_texts != null ? ' · ' + s.scored_texts + ' texts' : s.total ? ` · ${s.completed}/${s.total}` : ''), 'small'));
      b.onclick = () => { for (const x of list.children) x.classList.remove('on'); b.classList.add('on'); openRun(r.name); }; list.append(b);
    }
    const e = el('div', null, 'empty'); e.append(el('p', 'Pick a run.')); $('run-detail').replaceChildren(e);
  } catch (e) { list.replaceChildren(el('div', e.message, 'err')); }
}

async function openRun(name) {
  const box = $('run-detail'); box.replaceChildren(el('p', 'Loading…', 'meta'));
  try { archive = await api('/api/run?name=' + encodeURIComponent(name)); } catch (e) { box.replaceChildren(el('div', e.message, 'err')); return; }
  const sum = archive.summary;
  if (sum && sum.kind === 'benchmark') return renderBenchmark(sum, box);
  box.replaceChildren(); const h = el('h1', archive.name); h.style.cssText = 'font:500 16px var(--mono);margin-bottom:12px'; box.append(h);
  if (Array.isArray(sum)) {
    for (const v of sum) {
      const wrap = el('div', null, 'tablewrap tall'), t = el('table'), hr = el('tr');
      hr.append(th(v.name), th('Texts', '', 1), th('Detected', 'Share of texts detected', 1), th('Mean z', 'Average score', 1)); t.append(hr);
      for (const [c, s] of Object.entries(v.conditions)) { const tr = el('tr'), td = el('td', null, 'n'); td.append(heat(s.rate)); tr.append(el('td', c.replaceAll('_', ' ')), el('td', String(s.n), 'n'), td, el('td', fmt(s.mean_z), 'n')); t.append(tr); }
      wrap.append(t); box.append(wrap);
    }
  } else {
    const rows = Object.entries(sum || {}).filter(([, v]) => v && v.mean_z !== undefined).map(([n, v]) => ({ label: n.replaceAll('_', ' '), value: Math.max(v.mean_z, 0), text: fmt(v.mean_z, 1) + ' · ' + pct(v.detection_rate), color: v.mean_z >= 4 ? '#1b6c80' : '#cdced1' }));
    if (rows.length) box.append(card(el('h2', 'Mean score · share detected'), hbars(rows, { left: 150 })));
    if (archive.rows.length) {
      const wrap = el('div', null, 'tablewrap'), t = el('table'), hr = el('tr');
      hr.append(th('Prompt'), th('Condition'), th('Tokens', '', 1), th('Favored', '', 1), th('z', '', 1), th('Detected', '', 1)); t.append(hr);
      for (const r of archive.rows.slice(0, 200)) { const tr = el('tr'); [r.prompt_id || r.sample || '', (r.condition || '').replaceAll('_', ' '), r.scored_tokens, r.green_ratio === '' || r.green_ratio == null ? '—' : pct(Number(r.green_ratio)), fmt(r.z_score), yes(r.detected) ? 'Yes' : 'No'].forEach((v, i) => tr.append(el('td', String(v), i > 1 ? 'n' : ''))); t.append(tr); }
      wrap.append(t); box.append(wrap);
    }
  }
  const files = Object.keys(archive.texts || {});
  if (files.length) { const sel = el('select'); for (const f of files) sel.append(el('option', f)); const body = el('div', archive.texts[files[0]], 'hl'); sel.onchange = () => { body.textContent = archive.texts[sel.value]; }; box.append(card(sel, body)); }
}

/* ---------- wiring ---------- */
async function init() {
  const data = await api('/api/init'); token = data.token; $('key').value = data.key;
  for (const p of data.prompts.slice(0, 5)) {
    const text = typeof p === 'string' ? p : (p.prompt || p.text || '');
    const b = el('button', text.replace(/^Explain (how |why |the )?/i, '').replace(/\.$/, ''), null, { type: 'button' }); b.onclick = () => { $('prompt').value = text; }; $('prompt-chips').append(b);
  }
  if (data.prompts.length && typeof data.prompts[0] === 'string') $('prompt').value = data.prompts[0];
}
$('nav').onclick = e => { if (e.target.dataset.view) setView(e.target.dataset.view); };
$('menu').onclick = () => document.body.classList.toggle('nav-open');
for (const b of document.querySelectorAll('.seg button')) b.onclick = () => setMode(b.dataset.mode);
$('run').onclick = () => start(view);
$('run-exp').onclick = () => { mode = 'real'; start('experiment'); };
$('run-bench').onclick = () => { mode = 'real'; document.querySelector('.pop').open = false; start('benchmark'); };
$('refresh').onclick = loadRuns;
$('use-last').onclick = () => { if (last) $('text').value = last.text; };
$('newkey').onclick = async () => { $('key').value = (await api('/api/init')).key; };
$('cancel').onclick = () => jobId && api('/api/cancel', { id: jobId });
setMode('real');
init().catch(e => fail(e.message));
setView('results');
