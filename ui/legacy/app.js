const $ = id => document.getElementById(id);
let tab = 'generate', token = '', job = null, archive = null, ready = false;
let jobContext = null, activeResult = null, lastGenerated = null;
const results = {};
const copy = {
  generate: ['Generate & compare', 'Read two passages side by side: one written normally, one with a hidden watermark.', 'What should the AI write?'],
  detect: ['Check a passage', 'Look for the pattern made with your watermark key. This is not a general AI detector.', 'Choose what to check'],
  attack: ['Try rewriting', 'Rewrite a watermarked passage, then check whether the hidden pattern survives.', 'Choose a rewriting method'],
  experiment: ['Batch experiment', 'Test the watermark across multiple topics and rewritten versions.', 'Set up your experiment'],
  results: ['Saved experiments', 'Review what earlier experiments measured and read their saved samples.', '']
};
const configIds = ['mode', 'model', 'revision', 'key', 'gamma', 'delta', 'threshold', 'max_new_tokens', 'temperature', 'top_p', 'seed'];
async function api(url, data) {
  const response = await fetch(url, data ? {method:'POST', headers:{'Content-Type':'application/json', 'X-Dashboard-Token':token}, body:JSON.stringify(data)} : {});
  const value = await response.json();
  if (!response.ok) throw Error(value.error || 'The request could not be completed.');
  return value;
}
function error(e) { $('error').textContent = e.message; $('error').hidden = false; }
function modeNotice() {
  const offline = $('mode').value === 'offline';
  $('model-fields').hidden = offline;
  $('generation-fields').hidden = tab !== 'generate' || offline;
  $('intro').hidden = tab !== 'generate';
  $('text-label').textContent = offline ? 'Demo numbers to check' : 'Passage to check';
  $('text').placeholder = offline ? 'Generate a demo first, or paste its numbers here: 12 45 8 …' : 'Paste the passage you want to check…';
  $('text-help').textContent = offline
    ? 'Each number stands for a token (a piece of text). Use numbers from the quick demo, with the same key and settings.'
    : 'Use the same watermark key, model version, and favored token share used to generate this passage. Text from another AI will not necessarily contain this watermark.';
  const notices = {
    generate: offline
      ? 'Start here: click Run quick demo below. You will get two sequences of numbers, each standing for a token. A tiny untrained model makes them, so they are not readable sentences. The starting input is fixed; there is no prompt to enter.'
      : 'Write instructions below, then generate two readable samples. The first run may download the model and take longer. Some models require Hugging Face access.',
    detect: offline
      ? 'Check numbers from the quick demo. The detector rebuilds the favored choices using your key, then checks how often they appear.'
      : 'The detector checks for your particular word-choice pattern. It only needs the tokenizer (the model’s text-to-token rules), so it does not generate new text.',
    attack: offline
      ? 'Rewriting needs readable text. Choose “Readable text” above, generate a watermarked sample, then bring it here using “Try rewriting this”.'
      : 'Use a watermarked sample generated in readable-text mode. Rewording or translation changes token choices, which can weaken the pattern. Compare meaning and length as well as the score.',
    experiment: offline
      ? 'The full experiment needs readable text and language models. Choose “Readable text” above, or run the numeric batch demo below. The numeric demo tests token substitutions; it does not test real rewriting.'
      : 'This runs generation, three rounds of rewording, and translation to French and back. It may download several models. Start with one topic to check your setup.'
  };
  $('notice').textContent = notices[tab] || '';
  $('run').textContent = tab === 'generate' ? (offline ? 'Run quick demo' : 'Generate two samples') : tab === 'detect' ? 'Check for my watermark' : tab === 'attack' ? 'Rewrite and compare' : 'Run batch experiment';
  $('run').disabled = !ready || Boolean(job) || (offline && ['attack', 'experiment'].includes(tab));
  $('smoke').hidden = tab !== 'experiment' || !offline;
  $('settings').hidden = tab === 'experiment' && offline;
  $('delta').closest('label').hidden = !['generate', 'experiment'].includes(tab);
  $('action-help').textContent = offline ? (tab === 'generate' ? 'No setup or downloads needed. Results appear below.' : tab === 'detect' ? 'Checking uses the key in Advanced settings.' : 'Choose readable text above to enable this action.') : 'First use may download model files. Progress appears below.';
  $('use-last').hidden = !lastGenerated || lastGenerated.context.mode !== $('mode').value || !['detect', 'attack'].includes(tab);
  document.querySelectorAll('[data-generate-only]').forEach(n => n.hidden = tab === 'detect');
}
function switchTab(name) {
  tab = name;
  document.querySelectorAll('[data-tab]').forEach(b => { b.classList.toggle('selected', b.dataset.tab === tab); b.setAttribute('aria-current', b.dataset.tab === tab ? 'page' : 'false'); });
  $('heading').textContent = copy[tab][0]; $('description').textContent = copy[tab][1]; $('setup-heading').textContent = copy[tab][2];
  $('controls').hidden = tab === 'results'; $('results').hidden = tab !== 'results';
  $('text-fields').hidden = !['detect', 'attack'].includes(tab); $('attack-field').hidden = tab !== 'attack'; $('experiment-fields').hidden = tab !== 'experiment';
  $('intro').hidden = tab !== 'generate'; $('error').hidden = true; $('progress').hidden = true;
  $('navigation').classList.remove('open'); $('menu').setAttribute('aria-expanded', 'false');
  $('output').hidden = !results[tab] || tab === 'results'; activeResult = results[tab] || null;
  if (activeResult) render(activeResult);
  modeNotice();
  if (tab === 'results') refresh().catch(error);
}
function payload(action = tab) {
  const data = {action};
  for (const id of ['mode', 'model', 'revision', 'key', 'prompt', 'text', 'attack']) data[id] = $(id).value;
  for (const id of ['gamma', 'delta', 'threshold', 'max_new_tokens', 'temperature', 'top_p', 'seed', 'limit']) data[id] = Number($(id).value);
  return data;
}
function busy(value) {
  document.querySelectorAll('#controls input, #controls select, #controls textarea, #controls button, nav button').forEach(n => n.disabled = value);
  $('cancel').disabled = false; $('cancel').hidden = !value;
  $('cancel').textContent = 'Stop run';
  if (!value) modeNotice();
}
async function start(action) {
  if (!ready || job) return;
  $('error').hidden = true;
  if (['detect', 'attack'].includes(action) && !$('text').value.trim()) {
    error(Error('Add a sample first. Generate one, then choose “Check this sample”, or paste it into the passage box.')); $('text').focus(); return;
  }
  const data = payload(action);
  jobContext = {...data};
  busy(true); $('state').textContent = 'Starting'; $('progress').hidden = false;
  $('progress-heading').textContent = ({generate:'Generating two samples', detect:'Checking your passage', attack:'Rewriting and checking', experiment:'Running the batch experiment', smoke:'Running the numeric batch demo'})[action];
  $('progress-message').textContent = 'Starting the local job. ' + (data.mode === 'offline' ? 'No model downloads are needed.' : 'Model files may need to download on first use.');
  $('log').textContent = 'Starting…';
  try { job = (await api('/api/jobs', data)).id; poll(); }
  catch (e) { busy(false); $('state').textContent = 'Could not start'; $('progress-message').textContent = 'The run could not start. Check the message above and try again.'; error(e); }
}
async function poll() {
  try {
    const current = await api('/api/jobs/' + job);
    $('state').textContent = ({running:'Working', cancelling:'Stopping', cancelled:'Stopped', complete:'Finished', failed:'Needs attention'})[current.status] || current.status;
    $('log').textContent = current.log || 'Loading the local dependencies…';
    if (['running', 'cancelling'].includes(current.status)) {
      const log = current.log || '';
      $('progress-message').textContent = current.status === 'cancelling' ? 'Stopping this run. Please wait for the local job to close.'
        : /Generating watermarked/i.test(log) ? 'Writing the sample with the hidden pattern. We will compare it with the ordinary sample.'
        : /Generating control/i.test(log) ? 'Writing the ordinary sample first, without adding a watermark.'
        : /Paraphrase round/i.test(log) ? 'Rewording the passage and measuring the pattern in each version.'
        : /translation model/i.test(log) ? 'Preparing translation and checking the rewritten passage. First use may download translation models.'
        : jobContext.mode === 'real' ? 'Preparing the language model. First use may download files; details are in the log below.'
        : 'Preparing the tiny local model. No model downloads are needed.';
      setTimeout(poll, 1200); return;
    }
    job = null; busy(false);
    if (current.result?.samples) {
      const entry = {result:current.result, context:jobContext};
      results[tab] = entry; activeResult = entry;
      if (jobContext.action === 'generate') lastGenerated = entry;
      render(entry); modeNotice();
      $('progress-message').textContent = 'Finished. Your results and explanation are below.';
      $('output').scrollIntoView({behavior:'smooth', block:'start'});
    } else if (current.status !== 'failed' && current.status !== 'cancelled') {
      $('progress-message').textContent = 'Finished. Open Saved experiments to review the measurements.';
      const open = el('button', 'View saved experiment', 'secondary'); open.onclick = () => switchTab('results'); $('progress-message').append(' ', open);
    }
    if (current.status === 'cancelled') $('progress-message').textContent = 'Stopped. You can change the settings and start again.';
    if (current.status === 'failed') {
      $('progress-message').textContent = 'The run stopped before it finished. Expand the technical progress log for details.';
      const blockedDownload = /ProxyError|ConnectionError|LocalEntryNotFoundError|couldn.t connect|offline mode/i.test(current.log || '');
      error(Error(blockedDownload && jobContext.mode === 'real'
        ? 'The language model is not available locally, and its download could not connect. For the default Qwen model, run .\\download_model.ps1 in a normal PowerShell window with internet access, then generate again. For another model, pass -Model and -Revision to that script or enter its downloaded folder in Language model. The numeric test cannot produce paragraphs.'
        : 'This run could not finish. Check the technical progress log for the model or dependency error.'));
    }
  } catch (e) {
    $('state').textContent = 'Connection interrupted';
    $('progress-message').textContent = 'Could not reach the local server. The job may still be running. Checking again…';
    $('log').textContent = e.message;
    setTimeout(poll, 3000);
  }
}
function fmt(n, d = 2) { return n === null || n === undefined || n === '' ? '—' : Number(n).toFixed(d); }
function el(tag, text, cls) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; }
function useSample(entry, sample, destination) {
  for (const id of configIds) if (entry.context[id] !== undefined) $(id).value = entry.context[id];
  $('text').value = sample.text;
  switchTab(destination); $('text').focus();
}
function summarize(entry) {
  const {result, context} = entry, samples = result.samples;
  const box = $('takeaway'); box.replaceChildren();
  box.append(el('p', result.offline ? 'Quick demo results · numeric tokens from an untrained model' : 'Readable-text results · ' + context.model, 'result-context'));
  if (samples.Control && samples.Watermarked) {
    const ordinary = samples.Control.stats, marked = samples.Watermarked.stats;
    const expected = marked.detected && !ordinary.detected;
    box.append(el('h3', expected ? 'The detector found the added pattern.' : marked.detected ? 'Both samples crossed the cutoff.' : 'The added pattern did not cross the cutoff.'));
    box.append(el('p', 'Without watermark: score ' + fmt(ordinary.z_score) + '. With watermark: ' + fmt(marked.z_score) + '. The cutoff for this run is ' + marked.threshold + '.'));
    box.append(el('p', expected ? 'The marked sample contains enough favored choices to trigger this detector. The ordinary sample does not.' : ordinary.detected ? 'The ordinary sample also triggered the detector. That is a false positive here, because no watermark was added to it.' : 'A watermark was added, but this sample did not provide enough evidence for detection. A longer output or stronger watermark may help.'));
    box.append(el('p', result.offline ? 'These numbers are token IDs, not an error. This demo shows the mechanism with an untrained model. To generate sentences, choose readable text.' : 'Green highlights show favored token choices. Ordinary text can contain them too; their frequency is what matters.'));
    const next = el('button', result.offline ? 'Next: check the marked sample' : 'Next: try rewriting the marked sample', 'secondary');
    next.onclick = () => useSample(entry, samples.Watermarked, result.offline ? 'detect' : 'attack'); box.append(next);
  } else if (context.action === 'detect') {
    const stats = Object.values(samples)[0].stats;
    box.append(el('h3', stats.scored_tokens === 0 ? 'There is not enough text to check.' : stats.detected ? 'The pattern matches this key.' : 'No strong match for this key.'));
    box.append(el('p', 'Signal score: ' + fmt(stats.z_score) + '. Detection cutoff: ' + stats.threshold + '.'));
    box.append(el('p', stats.detected ? 'This passage crossed the detector’s cutoff. That is evidence of this watermark, not proof of authorship.' : 'This does not prove the passage is human-written or unwatermarked. A different key, different model settings, rewriting, or a short sample can all hide the signal.'));
  } else {
    const items = Object.entries(samples), before = items[0][1].stats, after = items[items.length - 1][1].stats;
    box.append(el('h3', after.detected ? 'The signal is still detectable after rewriting.' : 'The rewritten sample is below the cutoff.'));
    box.append(el('p', 'Original score: ' + fmt(before.z_score) + '. Final rewritten score: ' + fmt(after.z_score) + '. Cutoff: ' + after.threshold + '.'));
    if (!before.detected) box.append(el('p', 'The original was already below the cutoff, so this run cannot show that rewriting removed a detectable watermark.'));
    box.append(el('p', 'Read both versions for changes in meaning and length. A shorter passage supplies less evidence, which can reduce its score.'));
  }
}
function render(entry) {
  const {result} = entry; $('samples').replaceChildren(); $('output').hidden = false;
  summarize(entry);
  for (const [name, sample] of Object.entries(result.samples)) {
    const stats = sample.stats, card = el('article', undefined, 'sample'), head = el('div', undefined, 'sample-head');
    head.append(el('h3', name === 'Control' ? 'Without watermark' : name === 'Watermarked' ? 'With watermark' : name === 'Detected input' ? 'Your passage' : name), el('span', stats.scored_tokens === 0 ? 'Not enough tokens' : stats.detected ? 'Pattern detected' : 'Below cutoff', 'verdict' + (stats.detected ? ' detected' : '')));
    card.append(head);
    card.append(el('p', name === 'Control' ? 'Generated normally. Favored choices can still appear by chance.' : name === 'Watermarked' ? 'Generated with extra preference for the hidden pattern.' : 'Checked with the same watermark key and detection settings.', 'sample-explanation'));
    const metrics = el('div', undefined, 'metrics');
    for (const [label, value] of [['Signal score', fmt(stats.z_score)], ['Favored choices', stats.green_ratio === null ? fmt(null) : fmt(stats.green_ratio * 100, 1) + '%'], ['Checked tokens', stats.scored_tokens]]) {
      const metric = el('div'); metric.append(el('strong', String(value)), el('small', label)); metrics.append(metric);
    }
    if (!result.offline) {
      card.append(el('div', sample.text || 'The model returned no readable text. Try another prompt or a larger output limit.', 'sample-prose'));
      const copyText = el('button', 'Copy passage', 'text-button');
      copyText.onclick = async () => {
        try { await navigator.clipboard.writeText(sample.text); copyText.textContent = 'Passage copied'; }
        catch { copyText.textContent = 'Select the passage above to copy it'; }
      };
      card.append(copyText);
    }
    const tokenView = el('details', undefined, 'token-view');
    tokenView.append(el('summary', result.offline ? 'View numeric test tokens' : 'Inspect watermark highlights'));
    tokenView.append(el('p', 'Green = favored choice · plain = other choice · gray = excluded from the score.' + (result.offline ? '' : ' Token fragments are shown here for inspection; the complete decoded passage is above.'), 'help'));
    const text = el('div', undefined, 'tokens');
    for (const t of sample.tokens) { const span = el('span', t.text, t.status); span.title = t.status === 'green' ? 'Favored token: counted toward the pattern' : t.status === 'ignored' ? 'Excluded from the score (first, special, or repeated token pair)' : 'Other token choice'; text.append(span); }
    tokenView.append(text);
    if (result.offline) {
      card.append(el('p', 'This numeric test uses an untrained model, so it has no paragraph to display.', 'sample-note'));
      const readable = el('button', 'Generate readable paragraphs', 'secondary');
      readable.onclick = () => { $('mode').value = 'real'; switchTab('generate'); $('prompt').focus(); };
      card.append(readable);
    }
    card.append(tokenView, metrics);
    card.append(el('p', stats.scored_tokens === 0 ? 'No eligible token pairs were found, so there is no score to compare.' : 'Score ' + fmt(stats.z_score) + (stats.detected ? ' reaches ' : ' is below ') + 'the cutoff of ' + stats.threshold + '.', 'sample-note'));
    const details = el('details'), summary = el('summary', 'Technical measurements'); details.append(summary, el('p', 'Z-score: ' + fmt(stats.z_score) + '. Approximate p-value: ' + (stats.p_value_approx === null ? 'unavailable' : Number(stats.p_value_approx).toExponential(2)) + (sample.text_stats ? '. Score after converting the displayed text back into tokens: ' + fmt(sample.text_stats.z_score) + '.' : '.'), 'sample-note')); card.append(details);
    const actions = el('div', undefined, 'sample-actions');
    const check = el('button', 'Check this sample', 'secondary'); check.onclick = () => useSample(entry, sample, 'detect'); actions.append(check);
    if (!result.offline) { const rewrite = el('button', 'Try rewriting this', 'secondary'); rewrite.onclick = () => useSample(entry, sample, 'attack'); actions.append(rewrite); }
    card.append(actions); $('samples').append(card);
  }
  $('french').textContent = result.french ? 'French version used between translations: ' + result.french : '';
}
function download(name, body, type) { const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([body], {type})); link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); }
async function refresh() {
  const prior = $('runs').value, runs = await api('/api/runs'); $('runs').replaceChildren();
  for (const run of runs) { const option = el('option', (run.offline ? 'Numeric demo · ' : 'Language experiment · ') + run.name); option.value = run.name; $('runs').append(option); }
  if (runs.some(r => r.name === prior)) $('runs').value = prior;
  $('csv').disabled = !runs.length; $('runs').disabled = !runs.length;
  if (runs.length) await loadRun();
  else { archive = null; $('run-note').textContent = 'No experiments saved yet. Open Batch experiment to run one. Single sample comparisons stay in their own screen during this session.'; for (const id of ['rows','chart','passages','passage']) $(id).replaceChildren(); }
}
async function loadRun() {
  archive = await api('/api/run?name=' + encodeURIComponent($('runs').value));
  $('run-note').textContent = (archive.offline ? 'Numeric demo using synthetic token substitutions. This does not measure real paraphrasing or translation. ' : 'Language model experiment. ') + 'Run status: ' + archive.status.status + (archive.status.error ? ' · ' + archive.status.error : '');
  $('rows').replaceChildren();
  for (const row of archive.rows) { const tr = el('tr'); for (const value of [row.prompt_id || row.sample, row.condition.replaceAll('_',' '), row.scored_tokens, row.green_ratio == null ? fmt(null) : fmt(row.green_ratio * 100, 1) + '%', fmt(row.z_score), row.detected === true || row.detected === 'True' ? 'Yes' : 'No']) tr.append(el('td', String(value))); $('rows').append(tr); }
  $('chart').replaceChildren(el('h2', 'Average signal score'));
  const groups = Object.entries(archive.summary).filter(([, value]) => value && typeof value === 'object' && value.mean_z !== undefined);
  const max = Math.max(1, ...groups.map(([, value]) => Math.abs(value.mean_z || 0)));
  $('chart').append(el('p', 'Bars share a scale from −' + fmt(max) + ' to +' + fmt(max) + '. The center is zero. Counts show detected samples / checked samples.', 'chart-scale'));
  if (!groups.length) $('chart').append(el('p', 'Measurements will appear as samples are checked.'));
  for (const [name, value] of groups) {
    const row = el('div', undefined, 'chart-row'), track = el('div', undefined, 'chart-track'), fill = el('div', undefined, 'chart-fill');
    const score = value.mean_z || 0; fill.style.width = Math.abs(score) / max * 50 + '%'; fill.style.position = 'absolute'; fill.style.left = (score < 0 ? 50 - Math.abs(score) / max * 50 : 50) + '%'; track.append(fill);
    row.append(el('span', name.replaceAll('_',' '), 'chart-label'), track, el('span', fmt(value.mean_z) + ' · ' + value.detected + '/' + value.samples, 'chart-value')); $('chart').append(row);
  }
  $('passages').replaceChildren(); for (const name of Object.keys(archive.texts)) { const option = el('option', name); option.value = name; $('passages').append(option); } showPassage();
}
function showPassage() { $('passage').textContent = archive?.texts[$('passages').value] || 'This run has no saved passages. Numeric demos save token IDs instead of prose.'; }
document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => switchTab(button.dataset.tab));
$('run').onclick = () => start(tab); $('smoke').onclick = () => start('smoke');
$('cancel').onclick = async () => { try { if (job) { await api('/api/cancel', {id:job}); $('cancel').disabled = true; $('cancel').textContent = 'Stopping…'; } } catch (e) { error(e); } };
$('mode').onchange = () => { $('error').hidden = true; modeNotice(); };
$('menu').onclick = () => { const opened = $('navigation').classList.toggle('open'); $('menu').setAttribute('aria-expanded', String(opened)); };
$('use-last').onclick = () => useSample(lastGenerated, lastGenerated.result.samples.Watermarked, tab);
$('copy-key').onclick = async () => { try { await navigator.clipboard.writeText($('key').value); $('key-status').textContent = 'Key copied. Keep it somewhere private.'; } catch { $('key-status').textContent = 'Copy is unavailable. Select the key field and copy it manually.'; $('key').focus(); $('key').select(); } };
$('refresh').onclick = () => refresh().catch(error); $('runs').onchange = () => loadRun().catch(error); $('passages').onchange = showPassage;
$('presets').onchange = () => { if ($('presets').value) $('prompt').value = $('presets').value + ' Write a coherent passage of about 200 words.'; };
$('export').onclick = () => { if (activeResult) download('watermark-comparison.json', JSON.stringify(activeResult.result, null, 2), 'application/json'); };
$('csv').onclick = () => { if (!archive?.rows.length) return; const fields = Object.keys(archive.rows[0]); const escape = value => '"' + String(value ?? '').replaceAll('"','""') + '"'; download(archive.name + '.csv', [fields, ...archive.rows.map(row => fields.map(field => row[field]))].map(row => row.map(escape).join(',')).join('\r\n'), 'text/csv'); };
for (const id of ['max_new_tokens','temperature','top_p','seed']) $(id).closest('label').setAttribute('data-generate-only','');
api('/api/init').then(init => { token = init.token; $('key').value = init.key; for (const prompt of init.prompts) { const option = el('option', prompt); option.value = prompt; $('presets').append(option); } ready = true; switchTab('generate'); }).catch(e => { $('state').textContent = 'Server unavailable'; error(Error('Could not connect to Watermark Lab. Make sure the local server is running, then reload this page. ' + e.message)); });

