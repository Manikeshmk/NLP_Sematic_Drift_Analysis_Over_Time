/* ══════════════════════════════════════════════════════════════════════════
   app.js — Semantic Drift Analysis Dashboard
   Full client-side logic, API calls, and dynamic rendering
   ══════════════════════════════════════════════════════════════════════════ */

const API = "http://localhost:5000/api";
let loadedDecades = [];

// ─── On Load ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  await checkHealth();
  loadCaseStudies();
});

// ─── Health Check ─────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    loadedDecades = data.loaded_decades || [];

    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");

    if (loadedDecades.length > 0) {
      dot.className = "status-dot ok";
      text.textContent = `${loadedDecades.length} decades loaded (${loadedDecades[0]}–${loadedDecades[loadedDecades.length - 1]})`;
    } else {
      dot.className = "status-dot error";
      text.textContent = "No data found — download sgns/ dataset";
    }

    populateDecadeSelects();
  } catch (e) {
    const dot = document.getElementById("statusDot");
    dot.className = "status-dot error";
    document.getElementById("statusText").textContent = "API offline — run app.py";
  }
}

function populateDecadeSelects() {
  const selects = ["explorerYearA", "explorerYearB", "topYearA", "topYearB",
                   "constYearA",   "constYearB"];
  selects.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = "";
    loadedDecades.forEach(yr => {
      const opt = document.createElement("option");
      opt.value = yr;
      opt.textContent = yr;
      el.appendChild(opt);
    });
  });

  // sensible defaults
  const ya = document.getElementById("explorerYearA");
  const yb = document.getElementById("explorerYearB");
  if (ya && loadedDecades.length >= 2) {
    ya.value = loadedDecades[0];
    yb.value = loadedDecades[loadedDecades.length - 1];
  }
  const ta = document.getElementById("topYearA");
  const tb = document.getElementById("topYearB");
  if (ta && loadedDecades.length >= 2) {
    ta.value = loadedDecades[0];
    tb.value = loadedDecades[loadedDecades.length - 1];
  }
  const ca = document.getElementById("constYearA");
  const cb = document.getElementById("constYearB");
  if (ca && loadedDecades.length >= 2) {
    ca.value = loadedDecades[0];
    cb.value = loadedDecades[loadedDecades.length - 1];
  }
}

// ─── Tab Navigation ───────────────────────────────────────────────────────────

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(`tab-${btn.dataset.tab}`);
      if (panel) panel.classList.add("active");
    });
  });
}

// ─── Overlay & Toast ──────────────────────────────────────────────────────────

function showOverlay(msg = "Processing…") {
  document.getElementById("overlayMsg").textContent = msg;
  document.getElementById("overlay").classList.remove("hidden");
}

function hideOverlay() {
  document.getElementById("overlay").classList.add("hidden");
}

function toast(msg, type = "") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast ${type}`;
  setTimeout(() => el.classList.add("hidden"), 4000);
}

// ─── API Helpers ──────────────────────────────────────────────────────────────

async function api(path, body = null) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "GET" };
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

function b64Img(data, wrapper) {
  const el = document.getElementById(wrapper);
  el.innerHTML = `<img src="data:image/png;base64,${data}" alt="Chart" />`;
}

// ─── Word Explorer ────────────────────────────────────────────────────────────

function quickWord(word) {
  document.getElementById("explorerWord").value = word;
  runExplorer();
}

// ══════════════════════════════════════════════════════════════════════════════
// 🌌 DRIFT CONSTELLATION
// ══════════════════════════════════════════════════════════════════════════════

const CONST_PRESETS = {
  tech:   "The spread of artificial intelligence and computer networks transformed the broadcast of information.",
  idiom1: "Break the ice with cold silence between warm hearts.",
  idiom2: "The viral broadcast spread like disease across the network.",
  idiom3: "The network of political power controls the liberal economy.",
  idiom4: "Natural selection drives the evolution of artificial species.",
  idiom5: "Liberal democracy spread through the broadcast of enlightened reason.",
  idiom6: "Artificial intelligence simulates the natural network of human thought.",
};

function setConstPreset(key) {
  document.getElementById("constInput").value = CONST_PRESETS[key] || "";
}

// Word colour palette (must match Python side)
const CONST_PALETTE = [
  "#a855f7","#3b82f6","#22c55e","#f97316","#ef4444",
  "#facc15","#06b6d4","#ec4899","#84cc16","#8b5cf6",
  "#14b8a6","#f43f5e","#fb923c","#a3e635","#38bdf8",
];

let _lastConstellationData = null;   // for PNG download

async function runConstellation() {
  const text  = document.getElementById("constInput").value.trim();
  const yearA = parseInt(document.getElementById("constYearA").value);
  const yearB = parseInt(document.getElementById("constYearB").value);

  if (!text)          return toast("Enter a sentence or phrase first", "error");
  if (yearA >= yearB) return toast("From decade must be before To decade", "error");

  const btn = document.getElementById("constBtn");
  btn.disabled = true;
  showOverlay("Building drift constellation — loading all decades…");

  try {
    const data = await api("/sentence-constellation", { text, year_a: yearA, year_b: yearB });
    _lastConstellationData = data;
    renderConstellation(data);
    document.getElementById("constResults").classList.remove("hidden");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    hideOverlay();
    btn.disabled = false;
  }
}

// ─── Render All Constellation Components ─────────────────────────────────────

function renderConstellation(data) {
  const meta = data.meta;
  const words = meta.content_words;
  const ref = meta.reference_year;

  // ── Metric cards ─────────────────────────────────────────────────────────
  const sentScores  = meta.sentence_scores.filter(s => s !== null);
  const maxSent     = sentScores.length ? Math.max(...sentScores).toFixed(3) : "N/A";
  const finalSent   = sentScores.length ? sentScores[sentScores.length - 1].toFixed(3) : "N/A";
  const topWord     = meta.stability_rank[0];
  const bottomWord  = meta.stability_rank[meta.stability_rank.length - 1];

  document.getElementById("constMetrics").innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Words Tracked</div>
      <div class="metric-value accent">${words.length}</div>
      <div class="metric-sub">content words found</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Decades Spanned</div>
      <div class="metric-value accent">${meta.decades.length}</div>
      <div class="metric-sub">${meta.decades[0]} → ${meta.decades[meta.decades.length-1]}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Peak Sentence Drift</div>
      <div class="metric-value high">${maxSent}</div>
      <div class="metric-sub">from ${ref}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Final Drift (${meta.year_b})</div>
      <div class="metric-value mid">${finalSent}</div>
      <div class="metric-sub">sentence meaning shift</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Most Drifted Word</div>
      <div class="metric-value" style="color:#ef4444;font-size:1rem">${topWord ? topWord.word : 'N/A'}</div>
      <div class="metric-sub">${topWord ? topWord.max_drift.toFixed(3) : ''}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Most Stable Word</div>
      <div class="metric-value" style="color:#22c55e;font-size:1rem">${bottomWord ? bottomWord.word : 'N/A'}</div>
      <div class="metric-sub">${bottomWord ? bottomWord.max_drift.toFixed(3) : ''}</div>
    </div>
  `;

  // ── Interactive Plotly Chart ──────────────────────────────────────────────
  const plotDiv = document.getElementById("constellationPlot");
  if (typeof Plotly !== "undefined" && data.plotly) {
    Plotly.newPlot(plotDiv, data.plotly.data, data.plotly.layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      displaylogo: false,
      toImageButtonOptions: { format: "png", filename: "drift_constellation", scale: 2 },
    });
  } else {
    plotDiv.innerHTML = `<p style="color:#888;padding:20px">Plotly.js not loaded — ensure internet connection for interactive chart.</p>`;
  }

  // ── Correlation heatmap ───────────────────────────────────────────────────
  b64Img(data.corr_chart, "constCorrChart");

  // ── Static combined chart ─────────────────────────────────────────────────
  b64Img(data.static_chart, "constStaticChart");

  // ── Stability rank table ──────────────────────────────────────────────────
  renderConstellationRankTable(meta.stability_rank, meta.decades);
}

function renderConstellationRankTable(rank, decades) {
  if (!rank || !rank.length) {
    document.getElementById("constRankTable").innerHTML =
      `<p style="color:var(--text-muted)">No data</p>`;
    return;
  }

  const rows = rank.map((r, i) => {
    const color = CONST_PALETTE[i % CONST_PALETTE.length];
    const driftClass = r.max_drift > 0.4 ? "high" : r.max_drift > 0.2 ? "mid" : "low";
    const driftColor = r.max_drift > 0.4 ? "#ef4444" : r.max_drift > 0.2 ? "#facc15" : "#22c55e";

    // Sparkline: map scores to block chars
    const spark = (r.scores || []).map(s => {
      if (s === null) return '·';
      if (s < 0.1) return '▁';
      if (s < 0.2) return '▂';
      if (s < 0.3) return '▃';
      if (s < 0.4) return '▄';
      if (s < 0.5) return '▅';
      if (s < 0.6) return '▆';
      return '▇';
    }).join('');

    return `<tr>
      <td><span class="crt-swatch" style="background:${color}"></span></td>
      <td class="crt-word" style="color:${color}">${r.word}</td>
      <td class="crt-drift" style="color:${driftColor}">${r.max_drift.toFixed(4)}</td>
      <td>${r.max_drift > 0.35 ? '<span style="color:#ef4444;font-size:.75rem">⬤ High</span>'
             : r.max_drift > 0.15 ? '<span style="color:#facc15;font-size:.75rem">◐ Moderate</span>'
             : '<span style="color:#22c55e;font-size:.75rem">⬤ Stable</span>'}</td>
      <td class="crt-sparkline" title="Decade-by-decade drift">${spark}</td>
    </tr>`;
  }).join("");

  document.getElementById("constRankTable").innerHTML = `
    <table class="const-rank-table">
      <thead><tr>
        <th></th><th>Word</th><th>Peak Drift</th><th>Status</th>
        <th>Drift Arc 1800→1990</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── Download static PNG ──────────────────────────────────────────────────────

function downloadConstellationChart() {
  if (!_lastConstellationData) return toast("Run analysis first", "error");
  const b64 = _lastConstellationData.static_chart;
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${b64}`;
  a.download = "semantic_drift_constellation.png";
  a.click();
}
async function runExplorer() {
  const word = document.getElementById("explorerWord").value.trim().toLowerCase();
  const yearA = parseInt(document.getElementById("explorerYearA").value);
  const yearB = parseInt(document.getElementById("explorerYearB").value);

  if (!word) return toast("Enter a word first", "error");
  if (yearA >= yearB) return toast("From decade must be earlier than To decade", "error");

  showOverlay(`Analysing '${word}'…`);

  try {
    // Parallel: drift info + t-SNE + neighbour diagram
    const [drift, tsneData, nsChartData] = await Promise.all([
      api("/drift", { word, year_a: yearA, year_b: yearB }),
      api("/tsne", { word, year_a: yearA, year_b: yearB, top_n: 10 }),
      api("/neighbor-shift-chart", { word, year_a: yearA, year_b: yearB }),
    ]);

    renderExplorerResults(word, yearA, yearB, drift, tsneData, nsChartData);
    document.getElementById("explorerResults").classList.remove("hidden");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    hideOverlay();
  }
}

function renderExplorerResults(word, yearA, yearB, drift, tsneData, nsChartData) {
  // ── Metrics ──────────────────────────────────────────────────────────────
  const score = drift.drift_score;
  const scoreClass = score > 0.5 ? "high" : score > 0.25 ? "mid" : "low";

  const ns = drift.neighbor_shift;
  const jacc = ns.jaccard_similarity;

  document.getElementById("explorerMetrics").innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Drift Score</div>
      <div class="metric-value ${scoreClass}">${score.toFixed(3)}</div>
      <div class="metric-sub">Cosine distance</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Change Type</div>
      <div class="metric-value accent" style="font-size:1rem">${drift.change_type}</div>
      <div class="metric-sub">${yearA} → ${yearB}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Jaccard Overlap</div>
      <div class="metric-value ${jacc > 0.5 ? 'low' : jacc > 0.25 ? 'mid' : 'high'}">${jacc.toFixed(3)}</div>
      <div class="metric-sub">Neighbour similarity</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Gained Contexts</div>
      <div class="metric-value accent">${ns.gained.length}</div>
      <div class="metric-sub">New neighbours</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Lost Contexts</div>
      <div class="metric-value ${ns.lost.length > 5 ? 'mid' : 'low'}">${ns.lost.length}</div>
      <div class="metric-sub">Dropped neighbours</div>
    </div>
  `;

  // ── Neighbour Lists ───────────────────────────────────────────────────────
  const sharedSet = new Set(ns.shared);
  const gainedSet = new Set(ns.gained);

  function renderNbrList(containerId, neighbours, era, color) {
    const items = neighbours.map(([w, sim]) => {
      let badge = "";
      if (sharedSet.has(w)) badge = `<span class="nbr-badge shared-badge">shared</span>`;
      else if (gainedSet.has(w)) badge = `<span class="nbr-badge gained-badge">gained</span>`;
      else badge = `<span class="nbr-badge lost-badge">lost</span>`;
      return `<li class="nbr-item">
        <span class="nbr-word" style="color:${color}">${w}</span>
        ${badge}
        <span class="nbr-sim">${sim.toFixed(3)}</span>
      </li>`;
    }).join("");

    document.getElementById(containerId).innerHTML = `
      <div class="nbr-title" style="color:${color}">Context in ${era}</div>
      <ul class="nbr-list">${items}</ul>
    `;
  }

  renderNbrList("nbrsOld", ns.neighbors_old.slice(0, 10), yearA, "#ef4444");
  renderNbrList("nbrsNew", ns.neighbors_new.slice(0, 10), yearB, "#3b82f6");

  // ── Charts ────────────────────────────────────────────────────────────────
  b64Img(tsneData.image, "tsneChart");
  b64Img(nsChartData.image, "nsChart");

  // ── Case Study ────────────────────────────────────────────────────────────
  const cs = drift.case_study;
  if (cs) {
    document.getElementById("caseStudyCard").style.display = "block";
    document.getElementById("caseStudyBody").innerHTML = `
      <div class="cs-meta-row">
        <div class="cs-meta-item">
          <span class="cs-meta-label">Change Type</span>
          <span class="cs-meta-val">${cs.type}</span>
        </div>
        <div class="cs-meta-item">
          <span class="cs-meta-label">Period Studied</span>
          <span class="cs-meta-val">${cs.decade_a} → ${cs.decade_b}</span>
        </div>
      </div>
      <p class="cs-body">${cs.description}</p>
    `;
  } else {
    document.getElementById("caseStudyCard").style.display = "none";
  }
}

// ─── Timeline ─────────────────────────────────────────────────────────────────

function quickTimeline(word) {
  document.getElementById("timelineWord").value = word;
  runTimeline();
}

async function runTimeline() {
  const word = document.getElementById("timelineWord").value.trim().toLowerCase();
  if (!word) return toast("Enter a word", "error");

  showOverlay(`Building timeline for '${word}'…`);
  try {
    const data = await api("/drift-timeline-chart", { word });
    b64Img(data.image, "timelineChart");
    document.getElementById("timelineTitle").textContent =
      `Drift Timeline — '${word}' (ref: ${loadedDecades[0]})`;

    // Build table
    const rows = data.years.map((yr, i) => {
      const s = data.scores[i];
      const pct = s != null ? (s * 100).toFixed(1) : null;
      const bar = s != null ? `<div class="drift-bar-bg"><div class="drift-bar-fill" style="width:${pct}%"></div></div>` : "—";
      const present = data.present_in[i] ? "✓" : "✗";
      return `<tr>
        <td>${yr}</td>
        <td style="color:${data.present_in[i] ? 'var(--green)' : 'var(--red)'}">${present}</td>
        <td>${s != null ? s.toFixed(4) : "—"}</td>
        <td class="drift-bar-cell">${bar}</td>
      </tr>`;
    }).join("");

    document.getElementById("timelineTableBody").innerHTML = `
      <table class="decade-table">
        <thead><tr><th>Decade</th><th>Present</th><th>Drift Score</th><th>Drift Bar</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    document.getElementById("timelineResults").classList.remove("hidden");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    hideOverlay();
  }
}

// ─── Heatmap ──────────────────────────────────────────────────────────────────

const PRESETS = {
  tech: "network, computer, virus, artificial, digital, software, hardware",
  emotion: "awful, nice, terrible, wicked, sick, literally",
  society: "broadcast, marriage, race, social, liberal, democracy, revolution",
};

function setHeatmapPreset(key) {
  document.getElementById("heatmapWords").value = PRESETS[key];
}

async function runHeatmap() {
  const raw = document.getElementById("heatmapWords").value;
  const words = raw.split(",").map(w => w.trim().toLowerCase()).filter(Boolean).slice(0, 15);
  if (words.length < 2) return toast("Enter at least 2 words", "error");

  showOverlay("Building heatmap…");
  try {
    const data = await api("/heatmap", { words });
    b64Img(data.image, "heatmapChart");
    document.getElementById("heatmapResults").classList.remove("hidden");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    hideOverlay();
  }
}

// ─── Top Drifted ──────────────────────────────────────────────────────────────

async function runTopDrifted() {
  const yearA = parseInt(document.getElementById("topYearA").value);
  const yearB = parseInt(document.getElementById("topYearB").value);
  const topN = parseInt(document.getElementById("topN").value);
  if (yearA >= yearB) return toast("From decade must be earlier than To decade", "error");

  showOverlay("Ranking vocabulary — this may take a minute…");
  try {
    // Parallel bar + wordcloud + table data
    const [barData, wc, top] = await Promise.all([
      api("/top-drifted-chart", { year_a: yearA, year_b: yearB, top_n: topN }),
      api("/wordcloud", { year_a: yearA, year_b: yearB, top_n: 80 }),
      api("/top-drifted", { year_a: yearA, year_b: yearB, top_n: topN }),
    ]);

    b64Img(barData.image, "topBarChart");
    b64Img(wc.image, "topWordCloud");

    const rows = top.results.map((r, i) => `
      <tr>
        <td class="rank-num">#${i + 1}</td>
        <td class="rank-word">${r.word}</td>
        <td class="rank-drift">${r.drift.toFixed(4)}</td>
        <td><div class="drift-bar-bg"><div class="drift-bar-fill" style="width:${(r.drift * 100).toFixed(1)}%"></div></div></td>
      </tr>`).join("");

    document.getElementById("topTable").innerHTML = `
      <table class="rank-table">
        <thead><tr><th>#</th><th>Word</th><th>Drift Score</th><th>Bar</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    document.getElementById("topResults").classList.remove("hidden");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    hideOverlay();
  }
}

// ─── Concept Relations ────────────────────────────────────────────────────────

async function runConcepts() {
  const raw = document.getElementById("pairsInput").value.trim();
  if (!raw) return toast("Enter word pairs", "error");

  const pairs = raw.split("\n")
    .map(line => line.split(",").map(w => w.trim().toLowerCase()))
    .filter(p => p.length === 2 && p[0] && p[1])
    .slice(0, 5);

  if (!pairs.length) return toast("Enter valid pairs (word1, word2 per line)", "error");

  showOverlay("Tracking concept relations…");
  try {
    const data = await api("/concept-relation-chart", { pairs });
    b64Img(data.image, "conceptChart");
    document.getElementById("conceptResults").classList.remove("hidden");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    hideOverlay();
  }
}

// ─── Case Studies ─────────────────────────────────────────────────────────────

async function loadCaseStudies() {
  try {
    const data = await api("/case-studies");
    renderCaseStudies(data.case_studies);
  } catch (e) {
    // API not started yet
  }
}

function renderCaseStudies(cases) {
  const grid = document.getElementById("caseGrid");
  if (!grid) return;
  grid.innerHTML = cases.map(cs => `
    <div class="case-card">
      <div class="case-word">${cs.word}</div>
      <div class="case-type">${cs.type}</div>
      <div class="case-desc">${cs.description}</div>
      <div class="case-eras">${cs.decade_a} → ${cs.decade_b}</div>
      <button class="case-run-btn" onclick="runCaseStudy('${cs.word}', ${cs.decade_a}, ${cs.decade_b})">
        ▷ Run Analysis
      </button>
    </div>
  `).join("");
}

function runCaseStudy(word, yearA, yearB) {
  // Switch to explorer tab and run
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

  const btn = document.querySelector('[data-tab="explorer"]');
  btn.classList.add("active");
  document.getElementById("tab-explorer").classList.add("active");

  // Set values
  document.getElementById("explorerWord").value = word;
  if (document.getElementById("explorerYearA")) {
    document.getElementById("explorerYearA").value = yearA;
    document.getElementById("explorerYearB").value = yearB;
  }

  runExplorer();
}
