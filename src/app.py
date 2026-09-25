"""
app.py  -  Task 6 web interface, plus three upgrades:
  - a "Smart" model toggle (sentence embeddings instead of TF-IDF)
  - explainability (highlighted keywords / similar past tickets)
  - bulk CSV upload with downloadable results

Run:
    python src/app.py
Then open:
    http://127.0.0.1:5000

The Smart model and bulk-upload's Smart option require the bonus
dependencies and a one-time training step:
    pip install -r requirements.txt
    python src/train_embeddings.py
The Classic model and everything else works without that step.
"""

import io
import csv
import json as _json

import pandas as pd
from flask import Flask, render_template_string, request, jsonify, Response

from predict import load_artifacts, classify, classify_batch

app = Flask(__name__)
model, vectorizer, le = load_artifacts()

EXAMPLES = [
    "I can't log in, it keeps saying my password is wrong even after I reset it.",
    "The app crashed while I was uploading a file.",
    "My monthly report is missing several entries.",
    "Please update the email address on my account.",
    "Everything has been loading really slowly since this morning.",
]

CATEGORY_COLORS = {
    "Login Issue":       ("#2E5AAC", "#E8EEFB"),
    "Application Error": ("#B23A48", "#FBEAEC"),
    "Report":            ("#6A4C93", "#F0EAF7"),
    "Account Update":    ("#A66B00", "#FBF1DF"),
    "Performance":       ("#1F6F63", "#E4F1EE"),
}
DEFAULT_COLOR = ("#4B5563", "#EEF0F2")

with open("data/sample_bulk_tickets.csv") as f:
    SAMPLE_CSV = f.read()

MAX_BULK_ROWS = 300

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ticket Triage</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#F3F5F7; --surface:#FFFFFF; --ink:#12161C; --ink-soft:#5B6472; --ink-faint:#8B93A1;
    --accent:#1F6F63; --accent-ink:#0E4D44; --accent-soft:#E4F1EE; --border:#DDE2E7; --danger:#B23A48;
    --radius:10px;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;}
  body{background:var(--bg);color:var(--ink);font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:1040px;margin:0 auto;padding:56px 24px 80px;}
  header{margin-bottom:28px;max-width:640px;}
  h1{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:2.35rem;letter-spacing:-0.01em;margin:0 0 10px;}
  header p{margin:0;color:var(--ink-soft);font-size:1.02rem;max-width:52ch;}
  .status{display:inline-flex;align-items:center;gap:6px;font-size:.8rem;color:var(--accent-ink);background:var(--accent-soft);padding:4px 10px 4px 8px;border-radius:999px;margin-bottom:18px;font-weight:500;}
  .status .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);}

  .controls-row{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:22px;}
  .tabs{display:inline-flex;background:#E9ECEF;border-radius:9px;padding:3px;}
  .tab{border:none;background:transparent;padding:8px 16px;border-radius:7px;font-family:inherit;font-size:.88rem;font-weight:600;color:var(--ink-soft);cursor:pointer;}
  .tab.active{background:#fff;color:var(--ink);box-shadow:0 1px 2px rgba(18,22,28,.08);}
  .model-toggle{display:inline-flex;background:#E9ECEF;border-radius:9px;padding:3px;}
  .toggle-btn{border:none;background:transparent;padding:8px 14px;border-radius:7px;font-family:inherit;font-size:.85rem;font-weight:600;color:var(--ink-soft);cursor:pointer;}
  .toggle-btn.active{background:var(--ink);color:#fff;}
  .model-note{font-size:.78rem;color:var(--ink-faint);margin-top:6px;}

  .panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);}
  .grid{display:grid;grid-template-columns:1.15fr 1fr;gap:28px;align-items:start;}
  @media (max-width:820px){.grid{grid-template-columns:1fr;}}

  .input-panel{padding:24px 24px 22px;}
  label.field-label{display:block;font-weight:600;font-size:.92rem;margin-bottom:10px;}
  textarea{width:100%;min-height:132px;resize:vertical;font-family:inherit;font-size:.98rem;line-height:1.55;color:var(--ink);padding:14px 16px;border:1px solid var(--border);border-radius:8px;background:#FCFCFC;transition:border-color .15s ease, box-shadow .15s ease;}
  textarea:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);background:#fff;}
  .hint-row{display:flex;justify-content:space-between;align-items:center;margin-top:8px;min-height:18px;}
  .field-error{color:var(--danger);font-size:.83rem;font-weight:500;}
  .char-count{color:var(--ink-faint);font-size:.8rem;}
  .examples{margin-top:20px;}
  .examples-label{font-size:.83rem;color:var(--ink-soft);margin-bottom:10px;font-weight:500;}
  .chip-row{display:flex;flex-wrap:wrap;gap:8px;}
  .chip{border:1px solid var(--border);background:#fff;color:var(--ink-soft);font-size:.83rem;padding:6px 12px;border-radius:999px;cursor:pointer;transition:border-color .15s ease, color .15s ease, background .15s ease;font-family:inherit;}
  .chip:hover{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft);}
  .submit-row{margin-top:22px;display:flex;align-items:center;gap:14px;}
  button.primary{background:var(--ink);color:#fff;border:none;border-radius:8px;padding:12px 22px;font-size:.95rem;font-weight:600;font-family:inherit;cursor:pointer;display:inline-flex;align-items:center;gap:9px;transition:background .15s ease, transform .08s ease;}
  button.primary:hover{background:var(--accent-ink);}
  button.primary:active{transform:scale(.98);}
  button.primary:disabled{background:var(--ink-faint);cursor:default;}
  .spinner{width:14px;height:14px;border-radius:50%;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;display:none;animation:spin .7s linear infinite;}
  button.primary.loading .spinner{display:inline-block;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .kbd-hint{color:var(--ink-faint);font-size:.8rem;}
  kbd{font-family:inherit;font-size:.78rem;border:1px solid var(--border);border-bottom-width:2px;border-radius:4px;padding:1px 6px;background:#fafafa;}
  @media (max-width:480px){.kbd-hint{display:none;}}

  .result-panel{padding:24px;min-height:360px;display:flex;flex-direction:column;}
  .result-heading{font-weight:600;font-size:.92rem;margin-bottom:4px;}
  .empty-state{flex:1;display:flex;flex-direction:column;justify-content:center;align-items:flex-start;color:var(--ink-faint);padding-top:8px;}
  .empty-state svg{margin-bottom:14px;opacity:.55;}
  .empty-state p{margin:0;font-size:.92rem;max-width:34ch;color:var(--ink-soft);}
  .result-body{display:none;}
  .result-body.show{display:block;animation:fadein .35s ease;}
  @keyframes fadein{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:translateY(0);}}
  .badge{display:inline-flex;align-items:center;gap:8px;font-weight:600;font-size:1.05rem;padding:7px 14px;border-radius:999px;margin:6px 0 20px;}
  .badge .swatch{width:9px;height:9px;border-radius:50%;}
  .breakdown-label{font-size:.82rem;color:var(--ink-soft);font-weight:500;margin-bottom:10px;}
  .bar-row{display:grid;grid-template-columns:120px 1fr 42px;align-items:center;gap:10px;margin-bottom:8px;font-size:.83rem;}
  .bar-row .cat-name{color:var(--ink-soft);}
  .bar-track{height:8px;background:#EEF0F2;border-radius:999px;overflow:hidden;}
  .bar-fill{height:100%;border-radius:999px;width:0%;transition:width .6s cubic-bezier(.22,.9,.3,1);}
  .bar-pct{color:var(--ink-faint);text-align:right;}

  .explain-block{margin-top:20px;padding-top:16px;border-top:1px solid var(--border);}
  .explain-label{font-size:.82rem;color:var(--ink-soft);font-weight:500;margin-bottom:10px;}
  .highlighted-text{font-size:.92rem;line-height:1.6;background:#FAFAFA;border:1px solid var(--border);border-radius:8px;padding:12px 14px;}
  .highlighted-text mark{background:#FFE9A8;color:var(--ink);border-radius:3px;padding:0 2px;}
  .similar-item{border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:8px;font-size:.85rem;}
  .similar-item .sim-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}
  .similar-item .sim-cat{font-weight:600;font-size:.78rem;padding:2px 8px;border-radius:999px;}
  .similar-item .sim-pct{color:var(--ink-faint);font-size:.78rem;}
  .similar-item .sim-text{color:var(--ink-soft);}

  .response-block{margin-top:22px;padding-top:18px;border-top:1px solid var(--border);}
  .response-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
  .response-header .label{font-size:.82rem;color:var(--ink-soft);font-weight:500;}
  .copy-btn{border:1px solid var(--border);background:#fff;color:var(--ink-soft);font-size:.78rem;font-family:inherit;padding:4px 10px;border-radius:6px;cursor:pointer;transition:border-color .15s ease,color .15s ease;}
  .copy-btn:hover{border-color:var(--ink-soft);color:var(--ink);}
  .response-text{font-size:.92rem;color:var(--ink);line-height:1.55;background:#FAFAFA;border:1px solid var(--border);border-radius:8px;padding:14px 16px;}

  .error-box{background:#FBEAEC;border:1px solid #F0C9CE;color:var(--danger);border-radius:8px;padding:12px 14px;font-size:.87rem;white-space:pre-line;}

  .bulk-panel{padding:24px;}
  .bulk-panel .field-label{margin-bottom:6px;}
  .bulk-sub{color:var(--ink-soft);font-size:.88rem;margin:0 0 16px;}
  input[type=file]{font-family:inherit;font-size:.88rem;}
  .bulk-actions{display:flex;align-items:center;gap:14px;margin-top:18px;flex-wrap:wrap;}
  .link-btn{background:none;border:none;color:var(--accent-ink);font-family:inherit;font-size:.85rem;font-weight:600;cursor:pointer;padding:0;text-decoration:underline;}
  .bulk-results{margin-top:22px;}
  .table-wrap{max-height:420px;overflow:auto;border:1px solid var(--border);border-radius:8px;}
  table{border-collapse:collapse;width:100%;font-size:.83rem;}
  th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap;max-width:260px;overflow:hidden;text-overflow:ellipsis;}
  th{background:#FAFAFA;position:sticky;top:0;font-weight:600;}
  tr:last-child td{border-bottom:none;}
  .bulk-summary{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;}
  .bulk-summary .count{font-size:.85rem;color:var(--ink-soft);}

  footer{margin-top:44px;color:var(--ink-faint);font-size:.8rem;}
  .hidden{display:none !important;}
  @media (prefers-reduced-motion: reduce){*{animation-duration:.001ms !important; transition-duration:.001ms !important;}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="status"><span class="dot"></span>Model loaded and ready</div>
    <h1>Ticket Triage</h1>
    <p>Describe a support ticket in plain language and see the category it would be routed to, before an agent even opens it.</p>
  </header>

  <div class="controls-row">
    <div class="tabs" role="tablist">
      <button type="button" class="tab active" data-tab="single">Single Ticket</button>
      <button type="button" class="tab" data-tab="bulk">Bulk Upload</button>
    </div>
    <div>
      <div class="model-toggle">
        <button type="button" class="toggle-btn active" data-model="classic">Classic</button>
        <button type="button" class="toggle-btn" data-model="smart">Smart</button>
      </div>
      <div class="model-note" id="model-note">TF-IDF + Logistic Regression &mdash; shows the exact keywords behind each prediction.</div>
    </div>
  </div>

  <!-- SINGLE TICKET TAB -->
  <div id="panel-single">
    <div class="grid">
      <div class="panel input-panel">
        <label class="field-label" for="description">Ticket description</label>
        <textarea id="description" placeholder="e.g. I can't log into my account, it keeps saying my password is wrong."></textarea>
        <div class="hint-row">
          <span class="field-error" id="field-error"></span>
          <span class="char-count" id="char-count"></span>
        </div>
        <div class="examples">
          <div class="examples-label">Try an example</div>
          <div class="chip-row">
            {% for ex in examples %}
            <button type="button" class="chip" data-text="{{ ex }}">{{ ex[:38] }}{{ "…" if ex|length > 38 else "" }}</button>
            {% endfor %}
          </div>
        </div>
        <div class="submit-row">
          <button type="button" class="primary" id="submit-btn">
            <span class="spinner"></span><span class="btn-label">Classify ticket</span>
          </button>
          <span class="kbd-hint"><kbd>Ctrl</kbd>/<kbd>&#8984;</kbd> + <kbd>Enter</kbd> to classify</span>
        </div>
      </div>

      <div class="panel result-panel" id="result-panel">
        <div class="empty-state" id="empty-state">
          <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
          <p>Type a ticket on the left and click <strong>Classify ticket</strong> to see the predicted category, confidence, and a suggested reply.</p>
        </div>
        <div class="result-body" id="result-body"></div>
      </div>
    </div>
  </div>

  <!-- BULK UPLOAD TAB -->
  <div id="panel-bulk" class="hidden">
    <div class="panel bulk-panel">
      <label class="field-label">Upload a CSV of tickets</label>
      <p class="bulk-sub">The file needs a column named <code>ticket_description</code> (or any column with "description" in its name). Up to {{ max_rows }} rows are processed.</p>
      <input type="file" id="csv-input" accept=".csv">
      <div class="bulk-actions">
        <button type="button" class="primary" id="bulk-submit-btn">
          <span class="spinner"></span><span class="btn-label">Classify CSV</span>
        </button>
        <button type="button" class="link-btn" id="sample-csv-btn">Download a sample CSV to try</button>
      </div>
      <div class="field-error" id="bulk-error" style="margin-top:10px;"></div>

      <div class="bulk-results hidden" id="bulk-results">
        <div class="bulk-summary">
          <span class="count" id="bulk-count"></span>
          <button type="button" class="primary" id="download-results-btn" style="padding:8px 16px;font-size:.85rem;">Download results CSV</button>
        </div>
        <div class="table-wrap"><table id="bulk-table"></table></div>
      </div>
    </div>
  </div>

  <footer>Logistic Regression, trained on 178 labeled tickets across 5 categories &mdash; Classic uses TF-IDF, Smart uses sentence embeddings.</footer>
</div>

<script>
const CATEGORY_COLORS = {{ category_colors_json | safe }};
const SAMPLE_CSV = {{ sample_csv_json | safe }};
let currentModel = 'classic';

function escapeHtml(str){
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.getElementById('panel-single').classList.toggle('hidden', tab !== 'single');
    document.getElementById('panel-bulk').classList.toggle('hidden', tab !== 'bulk');
  });
});

/* ---------- model toggle ---------- */
const modelNote = document.getElementById('model-note');
document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentModel = btn.dataset.model;
    modelNote.textContent = currentModel === 'smart'
      ? 'Sentence embeddings + Logistic Regression — understands rephrased/synonym-heavy tickets, explained via similar past tickets.'
      : 'TF-IDF + Logistic Regression — shows the exact keywords behind each prediction.';
  });
});

/* ---------- single ticket ---------- */
const textarea = document.getElementById('description');
const submitBtn = document.getElementById('submit-btn');
const btnLabel = submitBtn.querySelector('.btn-label');
const fieldError = document.getElementById('field-error');
const charCount = document.getElementById('char-count');
const emptyState = document.getElementById('empty-state');
const resultBody = document.getElementById('result-body');

function updateCharCount(){
  const n = textarea.value.length;
  charCount.textContent = n > 0 ? n + ' characters' : '';
}
textarea.addEventListener('input', () => {
  updateCharCount();
  if (textarea.value.trim()) fieldError.textContent = '';
});

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    textarea.value = chip.dataset.text;
    updateCharCount();
    fieldError.textContent = '';
    textarea.focus();
  });
});

function renderExplanation(explanation){
  if (explanation.type === 'keywords'){
    return `<div class="explain-block">
      <div class="explain-label">Why this category? (highlighted keywords)</div>
      <div class="highlighted-text">${explanation.highlighted_html}</div>
    </div>`;
  }
  if (explanation.type === 'similar_tickets'){
    let items = explanation.items.map(item => {
      const [accent, tint] = CATEGORY_COLORS[item.category] || ['#4B5563', '#EEF0F2'];
      return `<div class="similar-item">
        <div class="sim-head">
          <span class="sim-cat" style="background:${tint};color:${accent}">${escapeHtml(item.category)}</span>
          <span class="sim-pct">${item.similarity.toFixed(0)}% similar</span>
        </div>
        <div class="sim-text">${escapeHtml(item.text)}</div>
      </div>`;
    }).join('');
    return `<div class="explain-block">
      <div class="explain-label">Why this category? (most similar past tickets)</div>
      ${items}
    </div>`;
  }
  return '';
}

function renderResult(data){
  const [accent, tint] = CATEGORY_COLORS[data.category] || ['#4B5563', '#EEF0F2'];
  let bars = '';
  data.breakdown.forEach(row => {
    const [c] = CATEGORY_COLORS[row.category] || ['#4B5563', '#EEF0F2'];
    bars += `<div class="bar-row">
      <span class="cat-name">${escapeHtml(row.category)}</span>
      <div class="bar-track"><div class="bar-fill" style="background:${c}" data-pct="${row.percent}"></div></div>
      <span class="bar-pct">${row.percent.toFixed(0)}%</span>
    </div>`;
  });

  resultBody.innerHTML = `
    <div class="result-heading">Predicted category</div>
    <div class="badge" style="background:${tint};color:${accent}">
      <span class="swatch" style="background:${accent}"></span>${escapeHtml(data.category)}
    </div>
    <div class="breakdown-label">Confidence across all categories</div>
    ${bars}
    ${renderExplanation(data.explanation)}
    <div class="response-block">
      <div class="response-header">
        <span class="label">Suggested reply</span>
        <button type="button" class="copy-btn" id="copy-btn">Copy</button>
      </div>
      <div class="response-text" id="response-text">${escapeHtml(data.response)}</div>
    </div>
  `;

  emptyState.style.display = 'none';
  resultBody.classList.remove('show');
  void resultBody.offsetWidth;
  resultBody.classList.add('show');

  requestAnimationFrame(() => {
    resultBody.querySelectorAll('.bar-fill').forEach(el => { el.style.width = el.dataset.pct + '%'; });
  });

  document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(data.response).then(() => {
      const btn = document.getElementById('copy-btn');
      btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
  });
}

function renderModelError(message){
  emptyState.style.display = 'none';
  resultBody.innerHTML = `<div class="result-heading">Smart model not ready</div>
    <div class="error-box">${escapeHtml(message)}</div>`;
  resultBody.classList.remove('show');
  void resultBody.offsetWidth;
  resultBody.classList.add('show');
}

async function classify(){
  const text = textarea.value.trim();
  if (!text){
    fieldError.textContent = 'Type a ticket description first';
    textarea.focus();
    return;
  }
  submitBtn.disabled = true;
  submitBtn.classList.add('loading');
  btnLabel.textContent = 'Classifying…';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: text, model: currentModel})
    });
    const data = await res.json();
    if (!res.ok){ renderModelError(data.error || 'Something went wrong.'); return; }
    renderResult(data);
  } catch (err) {
    fieldError.textContent = "Couldn't reach the classifier — try again.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('loading');
    btnLabel.textContent = 'Classify ticket';
  }
}

submitBtn.addEventListener('click', classify);
textarea.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter'){ e.preventDefault(); classify(); }
});
updateCharCount();

/* ---------- bulk upload ---------- */
const csvInput = document.getElementById('csv-input');
const bulkBtn = document.getElementById('bulk-submit-btn');
const bulkBtnLabel = bulkBtn.querySelector('.btn-label');
const bulkError = document.getElementById('bulk-error');
const bulkResults = document.getElementById('bulk-results');
const bulkTable = document.getElementById('bulk-table');
const bulkCount = document.getElementById('bulk-count');
let lastBulkRows = null;
let lastBulkColumns = null;

document.getElementById('sample-csv-btn').addEventListener('click', () => {
  const blob = new Blob([SAMPLE_CSV], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'sample_bulk_tickets.csv';
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
});

function toCsvValue(v){
  const s = (v === null || v === undefined) ? '' : String(v);
  if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

document.getElementById('download-results-btn').addEventListener('click', () => {
  if (!lastBulkRows) return;
  const lines = [lastBulkColumns.map(toCsvValue).join(',')];
  lastBulkRows.forEach(row => {
    lines.push(lastBulkColumns.map(c => toCsvValue(row[c])).join(','));
  });
  const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'ticket_predictions.csv';
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
});

bulkBtn.addEventListener('click', async () => {
  bulkError.textContent = '';
  bulkResults.classList.add('hidden');
  if (!csvInput.files.length){
    bulkError.textContent = 'Choose a CSV file first';
    return;
  }
  const fd = new FormData();
  fd.append('file', csvInput.files[0]);
  fd.append('model', currentModel);

  bulkBtn.disabled = true;
  bulkBtn.classList.add('loading');
  bulkBtnLabel.textContent = 'Classifying…';
  try {
    const res = await fetch('/bulk-predict', {method: 'POST', body: fd});
    const data = await res.json();
    if (!res.ok){ bulkError.textContent = data.error || 'Something went wrong.'; return; }

    lastBulkRows = data.rows;
    lastBulkColumns = data.columns;
    bulkCount.textContent = data.truncated
      ? `Showing first ${data.rows.length} of ${data.total_rows} rows (limit reached)`
      : `${data.rows.length} ticket${data.rows.length === 1 ? '' : 's'} classified`;

    let thead = '<tr>' + data.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr>';
    let tbody = data.rows.map(row =>
      '<tr>' + data.columns.map(c => `<td title="${escapeHtml(String(row[c] ?? ''))}">${escapeHtml(String(row[c] ?? ''))}</td>`).join('') + '</tr>'
    ).join('');
    bulkTable.innerHTML = thead + tbody;
    bulkResults.classList.remove('hidden');
  } catch (err) {
    bulkError.textContent = "Couldn't reach the classifier — try again.";
  } finally {
    bulkBtn.disabled = false;
    bulkBtn.classList.remove('loading');
    bulkBtnLabel.textContent = 'Classify CSV';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        examples=EXAMPLES,
        category_colors_json=_json.dumps(CATEGORY_COLORS),
        sample_csv_json=_json.dumps(SAMPLE_CSV),
        max_rows=MAX_BULK_ROWS,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True) or {}
    description = (payload.get("description") or "").strip()
    model_choice = payload.get("model") or "classic"
    if not description:
        return jsonify({"error": "description is required"}), 400

    try:
        result = classify(description, model_choice=model_choice,
                           classic_artifacts=(model, vectorizer, le))
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "The smart model couldn't run. Make sure "
                                  "you've run 'python src/train_embeddings.py' "
                                  "with an active internet connection at least once."}), 500


def _find_description_column(columns):
    for col in columns:
        if col.strip().lower() == "ticket_description":
            return col
    for col in columns:
        if "description" in col.strip().lower():
            return col
    return None


@app.route("/bulk-predict", methods=["POST"])
def bulk_predict_api():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Choose a CSV file first"}), 400

    model_choice = request.form.get("model") or "classic"
    file = request.files["file"]

    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({"error": "Couldn't read that file as a CSV."}), 400

    if df.empty:
        return jsonify({"error": "That CSV has no rows."}), 400

    col = _find_description_column(list(df.columns))
    if col is None:
        return jsonify({
            "error": "No ticket-description column found. Found columns: "
                     + ", ".join(df.columns) +
                     ". Rename one to 'ticket_description' and try again."
        }), 400

    total_rows = len(df)
    truncated = total_rows > MAX_BULK_ROWS
    if truncated:
        df = df.head(MAX_BULK_ROWS)

    descriptions = df[col].fillna("").astype(str).tolist()

    try:
        predictions = classify_batch(descriptions, model_choice=model_choice)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "The smart model couldn't run. Make sure "
                                  "you've run 'python src/train_embeddings.py' "
                                  "with an active internet connection at least once."}), 500

    df["predicted_category"] = [p["category"] for p in predictions]
    df["confidence_percent"] = [round(p["confidence"], 1) for p in predictions]

    df = df.fillna("")
    return jsonify({
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "total_rows": total_rows,
        "truncated": truncated,
    })


if __name__ == "__main__":
    app.run(debug=True)
