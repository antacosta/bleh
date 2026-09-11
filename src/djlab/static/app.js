"use strict";

// ---- tiny state ----------------------------------------------------------
const state = {
  root: null,
  songs: [], // full list from /api/library
  startingSongId: null,
  direction: "none",
  horizonMode: "count", // "count" | "duration"
  debugMode: false,
  lastPlan: null,
  analyzePollHandle: null,
};

const el = (id) => document.getElementById(id);

// ---- fetch helper ----------------------------------------------------------
async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
  } catch (err) {
    showToast("Network error talking to local server: " + err.message);
    throw err;
  }
  let body = null;
  try {
    body = await res.json();
  } catch (_) {
    // no body
  }
  if (!res.ok) {
    const msg = (body && body.error) || res.statusText;
    showToast(msg);
    throw new Error(msg);
  }
  return body;
}

let toastTimer = null;
function showToast(msg) {
  const t = el("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 5000);
}

// ---- library ----------------------------------------------------------
async function loadLibrary() {
  const path = el("libraryPath").value.trim();
  if (!path) return;
  const data = await api("/api/library", { method: "POST", body: JSON.stringify({ path }) });
  applyLibraryData(data);
  localStorage.setItem("djlab.libraryPath", path);
}

async function refreshLibrary() {
  const data = await api("/api/library");
  applyLibraryData(data);
}

function applyLibraryData(data) {
  state.root = data.root;
  state.songs = data.songs || [];
  if (state.root) el("libraryPath").value = state.root;
  renderLibrarySummary();
  renderSongList();
  updateAnalyzeAllButton();
}

function renderLibrarySummary() {
  const summary = el("librarySummary");
  if (!state.root) {
    summary.textContent = "No library loaded yet.";
    return;
  }
  const analyzed = state.songs.filter((s) => s.analyzed).length;
  summary.textContent = `${state.songs.length} song(s) in ${state.root} — ${analyzed} analyzed, ${state.songs.length - analyzed} not yet analyzed.`;
}

function songLabel(song) {
  if (song.artist && song.title) return `${song.artist} – ${song.title}`;
  return song.title || song.filename;
}

function renderSongList() {
  const list = el("songList");
  list.innerHTML = "";
  for (const song of state.songs) {
    const li = document.createElement("li");
    li.className = "song-row" + (song.id === state.startingSongId ? " selected" : "");
    li.dataset.songId = song.id;

    const main = document.createElement("div");
    main.className = "song-main";
    const title = document.createElement("div");
    title.className = "song-title";
    title.textContent = songLabel(song);
    const meta = document.createElement("div");
    meta.className = "song-meta";
    const metaBits = [`${Math.round(song.duration_sec)}s`];
    if (song.analyzed) {
      if (song.bpm) metaBits.push(`${song.bpm} BPM`);
      if (song.key) metaBits.push(song.key);
      if (song.genre) metaBits.push(song.genre);
    }
    meta.textContent = metaBits.join(" · ");
    main.appendChild(title);
    main.appendChild(meta);

    const badge = document.createElement("span");
    badge.className = "badge " + (song.analyzed ? "analyzed" : "not-analyzed");
    badge.textContent = song.analyzed ? "analyzed" : "not analyzed";

    li.appendChild(main);
    li.appendChild(badge);

    if (song.analyzed) {
      const infoBtn = document.createElement("button");
      infoBtn.className = "song-info-btn";
      infoBtn.title = "Inspect analysis details";
      infoBtn.textContent = "ⓘ";
      infoBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openSongDebugModal(song.id);
      });
      li.appendChild(infoBtn);
    }

    li.addEventListener("click", () => selectStartingSong(song.id));
    list.appendChild(li);
  }
}

function selectStartingSong(songId) {
  const song = state.songs.find((s) => s.id === songId);
  if (!song) return;
  if (!song.analyzed) {
    showToast(`"${songLabel(song)}" hasn't been analyzed yet -- analyze it first.`);
    return;
  }
  state.startingSongId = songId;
  el("startingSongDisplay").textContent = `Starting from: ${songLabel(song)}`;
  renderSongList();
}

function updateAnalyzeAllButton() {
  const missing = state.songs.filter((s) => !s.analyzed).length;
  const btn = el("analyzeAllBtn");
  btn.disabled = missing === 0;
  btn.textContent = missing === 0 ? "All songs analyzed" : `Analyze all missing (${missing})`;
}

// ---- analysis job ----------------------------------------------------------
async function startAnalyzeAll() {
  const result = await api("/api/analyze", { method: "POST", body: JSON.stringify({}) });
  if (!result.started) {
    showToast(result.reason || "Could not start analysis");
    return;
  }
  el("analyzeProgressWrap").hidden = false;
  pollAnalyzeStatus();
}

function pollAnalyzeStatus() {
  clearTimeout(state.analyzePollHandle);
  state.analyzePollHandle = setTimeout(async () => {
    const status = await api("/api/analyze/status");
    const pct = status.total ? Math.round((status.completed / status.total) * 100) : 100;
    el("analyzeProgressFill").style.width = pct + "%";
    el("analyzeProgressLabel").textContent = status.running
      ? `${status.completed}/${status.total} — ${status.current || ""}`
      : `Done (${status.completed}/${status.total})`;
    if (status.errors && status.errors.length) {
      showToast(`${status.errors.length} song(s) failed to analyze -- see server log.`);
    }
    if (status.running) {
      pollAnalyzeStatus();
    } else {
      await refreshLibrary();
      setTimeout(() => { el("analyzeProgressWrap").hidden = true; }, 2000);
    }
  }, 800);
}

// ---- song debug modal ----------------------------------------------------------
async function openSongDebugModal(songId) {
  const data = await api(`/api/songs/${encodeURIComponent(songId)}/debug`);
  const content = el("songDebugModalContent");
  content.innerHTML = "";

  const h = document.createElement("h3");
  h.textContent = `${data.artist || ""} ${data.title ? "– " + data.title : data.id}`.trim();
  content.appendChild(h);

  const rows = [
    ["Duration", `${Math.round(data.duration_sec)}s`],
    ["BPM", `${data.tempo.bpm ?? "unknown"} (confidence ${fmtPct(data.tempo.bpm_confidence)})`],
    ["Time signature", `${data.tempo.time_signature ?? "unknown"} (confidence ${fmtPct(data.tempo.time_signature_confidence)})`],
    ["Key", `${data.harmonic.key ?? "unknown"} (confidence ${fmtPct(data.harmonic.key_confidence)})`],
    ["Overall energy", `${data.energy.overall_energy} (trend ${data.energy.energy_trend}/s)`],
    ["Vocal density", data.vocals.vocal_density],
    ["Genre", data.genre ?? "unknown"],
  ];
  content.appendChild(kvTable(rows));

  const secH = document.createElement("h3");
  secH.textContent = `Structure (${data.structure.sections.length} section(s))`;
  content.appendChild(secH);
  const secTable = document.createElement("table");
  secTable.className = "component-table";
  secTable.innerHTML = "<tr><th>Section</th><th>Time</th><th>Label</th><th>Energy</th></tr>";
  for (const s of data.structure.sections) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${s.id}</td><td>${s.start}s–${s.end}s</td><td>${s.label || "—"}</td><td>${s.energy}</td>`;
    secTable.appendChild(tr);
  }
  content.appendChild(secTable);

  const djH = document.createElement("h3");
  djH.textContent = "DJ affordances";
  content.appendChild(djH);
  content.appendChild(kvTable([
    ["Mix-in points", data.dj_affordances.mix_in_points.length],
    ["Mix-out points", data.dj_affordances.mix_out_points.length],
    ["Loop candidates", data.dj_affordances.loop_candidates.length],
  ]));

  el("songDebugModalBackdrop").hidden = false;
}

function kvTable(rows) {
  const table = document.createElement("table");
  table.className = "component-table";
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<th>${k}</th><td>${v}</td>`;
    table.appendChild(tr);
  }
  return table;
}

function fmtPct(x) { return Math.round((x || 0) * 100) + "%"; }

// ---- plan generation ----------------------------------------------------------
async function generatePlan() {
  if (!state.startingSongId) {
    showToast("Select a starting song first.");
    return;
  }
  const payload = { starting_song_id: state.startingSongId, direction: state.direction };
  if (state.horizonMode === "count") {
    payload.num_songs = parseInt(el("numSongsInput").value, 10) || 4;
  } else {
    payload.duration_minutes = parseFloat(el("durationInput").value) || 15;
  }

  const btn = el("generatePlanBtn");
  btn.disabled = true;
  btn.textContent = "Generating…";
  try {
    const plan = await api("/api/plan", { method: "POST", body: JSON.stringify(payload) });
    state.lastPlan = plan;
    renderPlan(plan);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Plan";
  }
}

function renderPlan(plan) {
  el("planArea").hidden = false;

  const summary = el("sequenceSummary");
  summary.innerHTML = "";
  const scorePill = document.createElement("span");
  scorePill.className = "score-pill big";
  scorePill.textContent = `Path score ${plan.sequence_score.path_score.toFixed(1)}`;
  const confPill = document.createElement("span");
  confPill.className = "score-pill";
  confPill.textContent = `Confidence ${fmtPct(plan.sequence_score.path_confidence)}`;
  const horizonPill = document.createElement("span");
  horizonPill.className = "score-pill";
  horizonPill.textContent = `${plan.achieved_horizon}/${plan.requested_horizon} songs planned`;
  summary.appendChild(scorePill);
  summary.appendChild(confPill);
  summary.appendChild(horizonPill);
  if (plan.excluded_unanalyzed_count > 0) {
    const excl = document.createElement("span");
    excl.className = "score-pill";
    excl.textContent = `${plan.excluded_unanalyzed_count} un-analyzed song(s) excluded`;
    summary.appendChild(excl);
  }

  const reasonsRow = document.createElement("div");
  reasonsRow.className = "reasons-row";
  for (const reason of plan.major_reasons) {
    const chip = document.createElement("span");
    chip.className = "reason-chip";
    chip.textContent = reason;
    reasonsRow.appendChild(chip);
  }
  summary.appendChild(reasonsRow);

  renderEnergyTrajectory(plan.sequence_score.energy_levels);
  renderSteps(plan);
  renderDebugPanel(plan);

  el("planArea").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderEnergyTrajectory(levels) {
  const wrap = el("energyTrajectory");
  wrap.innerHTML = "";
  for (let i = 0; i < levels.length; i++) {
    const barWrap = document.createElement("div");
    barWrap.className = "energy-bar-wrap";
    const bar = document.createElement("div");
    bar.className = "energy-bar";
    bar.style.height = Math.max(2, Math.round(levels[i] * 60)) + "px";
    const label = document.createElement("span");
    label.className = "energy-bar-label";
    label.textContent = i === 0 ? "start" : `#${i}`;
    barWrap.appendChild(bar);
    barWrap.appendChild(label);
    wrap.appendChild(barWrap);
  }
}

function renderSteps(plan) {
  const list = el("planSteps");
  list.innerHTML = "";

  for (let i = 0; i < plan.steps.length; i++) {
    const step = plan.steps[i];
    const transition = plan.transitions[i];

    if (i === 0) {
      list.appendChild(transitionRow(transition));
    }

    const li = document.createElement("li");
    li.className = "plan-step-card";

    const head = document.createElement("div");
    head.className = "plan-step-head";
    const title = document.createElement("div");
    title.innerHTML = `<div class="plan-step-title">${step.position}. ${step.song_id}</div>` +
      `<div class="plan-step-meta">${step.artist || ""}${step.genre ? " · " + step.genre : ""} · energy ${step.overall_energy}</div>`;
    const scores = document.createElement("div");
    scores.className = "plan-step-scores";
    scores.innerHTML =
      `<span class="score-pill">score ${step.transition_score.total_score.toFixed(1)}</span>` +
      `<span class="score-pill">conf ${fmtPct(step.transition_score.confidence)}</span>`;
    head.appendChild(title);
    head.appendChild(scores);
    li.appendChild(head);

    if (state.debugMode) {
      li.appendChild(componentTable(step.transition_score));
    }

    list.appendChild(li);

    if (i < plan.steps.length - 1) {
      list.appendChild(transitionRow(plan.transitions[i + 1]));
    }
  }
}

function transitionRow(transition) {
  const row = document.createElement("li");
  row.className = "transition-row";
  const pill = document.createElement("span");
  pill.className = "transition-pill";
  pill.textContent = "Transition: not yet implemented";
  row.appendChild(pill);
  if (state.debugMode && transition && transition.available_signal) {
    const sig = transition.available_signal;
    const detail = document.createElement("span");
    detail.textContent = `(available signal for future use: structure affordance ${sig.structure_affordance_score.toFixed(1)}, confidence ${fmtPct(sig.structure_affordance_confidence)})`;
    row.appendChild(detail);
  }
  return row;
}

const COMPONENT_NAMES = ["tempo_score", "harmonic_score", "energy_score", "rhythm_score", "structure_score", "style_score", "familiarity_score", "repetition_penalty"];

function componentTable(transitionScore) {
  const table = document.createElement("table");
  table.className = "component-table";
  table.innerHTML = "<tr><th>Component</th><th>Value</th><th>Confidence</th></tr>";
  for (const name of COMPONENT_NAMES) {
    const c = transitionScore[name];
    if (!c) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name.replace("_score", "").replace("_penalty", "")}</td><td>${c.value.toFixed(1)}</td><td>${fmtPct(c.confidence)}</td>`;
    table.appendChild(tr);
  }
  return table;
}

function renderDebugPanel(plan) {
  const panel = el("debugPanel");
  panel.hidden = !state.debugMode;
  if (!state.debugMode) return;

  const seqBlock = el("debugSequenceScore");
  seqBlock.innerHTML = "";
  const seqTable = document.createElement("table");
  seqTable.className = "component-table";
  seqTable.innerHTML = "<tr><th>Component</th><th>Value</th><th>Confidence</th></tr>";
  for (const [name, c] of Object.entries(plan.sequence_score.components)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td><td>${c.value.toFixed(1)}</td><td>${fmtPct(c.confidence)}</td>`;
    seqTable.appendChild(tr);
  }
  seqBlock.appendChild(seqTable);

  const lvgBlock = el("debugLocalVsGlobal");
  lvgBlock.innerHTML = "";
  const lvgTable = document.createElement("table");
  lvgTable.className = "component-table";
  lvgTable.innerHTML = "<tr><th>Opening song</th><th>Immediate score</th><th>Downstream path score</th><th>Depth reached</th><th></th></tr>";
  for (const c of plan.local_vs_global) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c.song_id}</td><td>${c.immediate_transition_score.toFixed(1)}</td>` +
      `<td>${c.downstream_path_score.toFixed(1)}</td><td>${c.depth_reached}</td>` +
      `<td>${c.chosen ? "← chosen" : ""}</td>`;
    lvgTable.appendChild(tr);
  }
  lvgBlock.appendChild(lvgTable);

  const notesBlock = el("debugSearchNotes");
  notesBlock.innerHTML = "";
  if (plan.search_notes && plan.search_notes.length) {
    const h = document.createElement("h3");
    h.textContent = "Search notes";
    notesBlock.appendChild(h);
    // The engine can log the same constrained-pool note once per beam node
    // that hit it; de-duplicate for display only -- this is a presentation
    // concern, not a change to what the planner reports.
    for (const note of new Set(plan.search_notes)) {
      const div = document.createElement("div");
      div.className = "debug-explanation";
      div.textContent = note;
      notesBlock.appendChild(div);
    }
  }
}

// ---- wiring ----------------------------------------------------------
function init() {
  el("loadLibraryBtn").addEventListener("click", loadLibrary);
  el("libraryPath").addEventListener("keydown", (e) => { if (e.key === "Enter") loadLibrary(); });
  el("analyzeAllBtn").addEventListener("click", startAnalyzeAll);
  el("generatePlanBtn").addEventListener("click", generatePlan);

  document.querySelectorAll("#directionButtons button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#directionButtons button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.direction = btn.dataset.direction;
    });
  });

  document.querySelectorAll("#horizonModeButtons button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#horizonModeButtons button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.horizonMode = btn.dataset.mode;
      el("numSongsInput").hidden = state.horizonMode !== "count";
      el("durationInput").hidden = state.horizonMode !== "duration";
    });
  });

  el("debugToggle").addEventListener("change", (e) => {
    state.debugMode = e.target.checked;
    if (state.lastPlan) renderPlan(state.lastPlan);
  });

  el("songDebugModalClose").addEventListener("click", () => { el("songDebugModalBackdrop").hidden = true; });
  el("songDebugModalBackdrop").addEventListener("click", (e) => {
    if (e.target === el("songDebugModalBackdrop")) el("songDebugModalBackdrop").hidden = true;
  });

  const savedPath = localStorage.getItem("djlab.libraryPath");
  if (savedPath) el("libraryPath").value = savedPath;

  refreshLibrary().catch(() => {});
}

document.addEventListener("DOMContentLoaded", init);
