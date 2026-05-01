/**
 * AI Multi-Resume Ranking Agent — Frontend Logic
 */
(function () {
  'use strict';

  const MAX_FILES = 5;
  const ALLOWED = ['.pdf', '.docx', '.doc'];
  const MAX_MB = 10;

  let selectedFiles = [];

  // ── DOM refs ──────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const uploadSection = $('#upload-section');
  const loadingSection = $('#loading-section');
  const resultsSection = $('#results-section');
  const jdTextarea = $('#jd-text');
  const dropZone = $('#drop-zone');
  const fileInput = $('#file-input');
  const fileList = $('#file-list');
  const submitBtn = $('#submit-btn');
  const loadingStep = $('#loading-step');
  const resultsContainer = $('#results-container');
  const summaryBanner = $('#summary-banner');
  const resetBtn = $('#reset-btn');
  const stepDots = document.querySelectorAll('.step-dot');

  // ── Drag & Drop ───────────────────────────────────────────
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    addFiles(Array.from(e.dataTransfer.files));
  });
  fileInput.addEventListener('change', () => { addFiles(Array.from(fileInput.files)); fileInput.value = ''; });

  function addFiles(files) {
    for (const f of files) {
      if (selectedFiles.length >= MAX_FILES) { toast(`Maximum ${MAX_FILES} resumes allowed.`); break; }
      const ext = '.' + f.name.split('.').pop().toLowerCase();
      if (!ALLOWED.includes(ext)) { toast(`Unsupported file: ${f.name}`); continue; }
      if (f.size > MAX_MB * 1024 * 1024) { toast(`${f.name} exceeds ${MAX_MB} MB limit.`); continue; }
      if (selectedFiles.some(s => s.name === f.name)) continue;
      selectedFiles.push(f);
    }
    renderFileList();
  }

  function renderFileList() {
    fileList.innerHTML = '';
    selectedFiles.forEach((f, i) => {
      const div = document.createElement('div');
      div.className = 'file-item';
      div.innerHTML = `
        <div class="file-info">
          <span class="file-icon">📄</span>
          <span class="file-name">${f.name}</span>
          <span class="file-size">${(f.size / 1024).toFixed(0)} KB</span>
        </div>
        <button class="remove-btn" data-idx="${i}" title="Remove">✕</button>`;
      div.querySelector('.remove-btn').addEventListener('click', () => { selectedFiles.splice(i, 1); renderFileList(); });
      fileList.appendChild(div);
    });
    submitBtn.disabled = selectedFiles.length === 0;
  }

  // ── Submit Flow ───────────────────────────────────────────
  submitBtn.addEventListener('click', async () => {
    const jdText = jdTextarea.value.trim();
    if (!jdText) { toast('Please enter a job description.'); return; }
    if (selectedFiles.length === 0) { toast('Please add at least one resume.'); return; }

    showView('loading');
    setLoadingStep('Uploading documents…');

    try {
      // Step 1: Upload
      const form = new FormData();
      form.append('job_description', jdText);
      selectedFiles.forEach(f => form.append('resumes', f));

      const upRes = await fetch('/upload', { method: 'POST', body: form });
      if (!upRes.ok) { const e = await upRes.json(); throw new Error(e.detail || 'Upload failed'); }
      const upData = await upRes.json();

      setLoadingStep(`Extracted ${upData.resumes_loaded} resume(s). Ranking candidates…`);

      // Step 2: Rank
      const rkRes = await fetch('/rank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: upData.session_id }),
      });
      if (!rkRes.ok) { const e = await rkRes.json(); throw new Error(e.detail || 'Ranking failed'); }
      const rkData = await rkRes.json();

      renderResults(rkData);
      showView('results');
    } catch (err) {
      toast(err.message, true);
      showView('upload');
    }
  });

  // ── View Switching ────────────────────────────────────────
  function showView(v) {
    uploadSection.style.display = v === 'upload' ? 'flex' : 'none';
    loadingSection.style.display = v === 'loading' ? 'block' : 'none';
    resultsSection.style.display = v === 'results' ? 'block' : 'none';
    stepDots.forEach((d, i) => {
      d.className = 'step-dot';
      if (v === 'upload') { if (i === 0) d.classList.add('active'); }
      else if (v === 'loading') { if (i === 0) d.classList.add('done'); if (i === 1) d.classList.add('active'); }
      else if (v === 'results') { d.classList.add('done'); if (i === 2) d.classList.add('active'); }
    });
  }

  function setLoadingStep(txt) { loadingStep.textContent = txt; }

  // ── Reset ─────────────────────────────────────────────────
  resetBtn.addEventListener('click', () => {
    selectedFiles = [];
    jdTextarea.value = '';
    renderFileList();
    resultsContainer.innerHTML = '';
    showView('upload');
  });

  // ── Render Results ────────────────────────────────────────
  function renderResults(data) {
    summaryBanner.innerHTML = `
      <span class="role-tag">${data.job_role}</span> — ${data.total_candidates} candidate(s) evaluated
      <br>${data.ranking_summary}`;

    resultsContainer.innerHTML = '';
    data.candidates.forEach(c => {
      const card = document.createElement('div');
      card.className = 'candidate-card';
      const rankClass = c.rank <= 3 ? `rank-${c.rank}` : 'rank-other';
      const circumference = 2 * Math.PI * 27;
      const offset = circumference - (c.score / 100) * circumference;

      card.innerHTML = `
        <!-- Header -->
        <div class="card-header">
          <div class="rank-badge ${rankClass}">#${c.rank}</div>
          <div class="candidate-info">
            <div class="candidate-name">${c.name}</div>
            <div class="candidate-file">${c.filename}</div>
          </div>
          <div class="score-gauge">
            <svg width="64" height="64" viewBox="0 0 64 64">
              <defs><linearGradient id="gaugeGrad${c.rank}" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#7c5cfc"/><stop offset="100%" stop-color="#4ea8ff"/>
              </linearGradient></defs>
              <circle class="bg-ring" cx="32" cy="32" r="27"/>
              <circle class="fg-ring" cx="32" cy="32" r="27"
                stroke="url(#gaugeGrad${c.rank})"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${circumference}"
                data-target="${offset}"/>
            </svg>
            <div class="score-text">${c.score.toFixed(0)}</div>
          </div>
        </div>


        <!-- Strengths / Weaknesses -->
        <div class="eval-grid">
          <div class="eval-list">
            <h4>✅ Strengths</h4>
            <ul>${c.strengths.map(s => `<li><span class="icon">✓</span> ${s}</li>`).join('')}</ul>
          </div>
          <div class="eval-list">
            <h4>⚠️ Areas to Improve</h4>
            <ul>${c.weaknesses.map(w => `<li><span class="icon">–</span> ${w}</li>`).join('')}</ul>
          </div>
        </div>

        <!-- Explanation -->
        <div class="explanation">${c.explanation}</div>`;

      resultsContainer.appendChild(card);
    });

    // Animate score gauges
    requestAnimationFrame(() => {
      document.querySelectorAll('.fg-ring').forEach(ring => {
        ring.style.strokeDashoffset = ring.dataset.target;
      });
      document.querySelectorAll('.score-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.w;
      });
    });
  }

  function scorebar(label, val) {
    return `<div class="score-bar-item">
      <label>${label}</label>
      <div class="score-bar-track"><div class="score-bar-fill" style="width:0" data-w="${val}%"></div></div>
      <div class="score-bar-val">${val.toFixed(0)}</div>
    </div>`;
  }

  // ── Toast ─────────────────────────────────────────────────
  function toast(msg, isError = true) {
    const el = document.createElement('div');
    el.className = `toast ${isError ? '' : 'success'}`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4500);
  }
})();
