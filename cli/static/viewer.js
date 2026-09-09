/* Report Room — client for the local report API.
   Three routes: #/ (the tape), #/digest (one printable sheet), #/run/<id>. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const THEME_KEY = "reportroom.theme";
const CALLS = [
  { key: "all", label: "All" },
  { key: "bull", label: "Buy" },
  { key: "flat", label: "Hold" },
  { key: "bear", label: "Sell" },
];

const state = {
  root: "",
  runs: [],
  run: null,          // { summary, sections } of the open run
  filter: "",
  call: "all",
  sort: "newest",
  expanded: new Set(),
  expandAll: false,
  matches: [],
  matchAt: -1,
  tapeScroll: 0,
  lastFetch: 0,
};

/* ── helpers ─────────────────────────────────────────────────────────── */

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

const num = (value, digits = 2) =>
  value === null || value === undefined
    ? null
    : value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

const pct = (value) =>
  value === null || value === undefined ? "" : `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;

const words = (n) => (n || 0).toLocaleString();

function stampOf(run) {
  const date = run.run_date || (run.generated_at || "").slice(0, 10);
  return [date, run.run_time].filter(Boolean).join(" · ") || run.run_id;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 1900);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme-label").textContent = theme === "ink" ? "Paper" : "Ink";
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
}

/* Price levels shared by the tape rows and the reader bar. */
function levelCells(run) {
  const cells = [];
  if (run.entry_price !== null) cells.push(["Entry", num(run.entry_price), ""]);
  if (run.stop_loss !== null) cells.push(["Stop", num(run.stop_loss), pct(run.stop_pct)]);
  if (run.price_target !== null) cells.push(["Target", num(run.price_target), pct(run.target_pct)]);
  return cells
    .map(([key, value, note]) => {
      const dir = note.startsWith("+") ? "up" : note.startsWith("-") ? "down" : "";
      return `<div><dt>${key}</dt><dd>${value}${note ? ` <em class="${dir}">${note}</em>` : ""}</dd></div>`;
    })
    .join("");
}

function horizonCell(run) {
  if (!run.time_horizon) return "";
  return `<div><dt>Horizon</dt><dd class="thin">${esc(run.time_horizon)}</dd></div>`;
}

function badge(run) {
  const call = run.action || run.rating;
  if (!call) return `<span class="badge badge-none">No call</span>`;
  return `<span class="badge">${esc(call)}</span>`;
}

/* ── data ────────────────────────────────────────────────────────────── */

async function loadRuns({ quiet = false } = {}) {
  const response = await fetch("/api/runs");
  const data = await response.json();
  state.root = data.root;
  state.runs = data.runs;
  state.lastFetch = Date.now();
  $("#root-path").textContent = data.root;
  $("#root-path").title = data.root;
  renderStrip();
  renderChips();
  renderTape();
  renderSheet();
  if (!quiet) document.title = `Report Room · ${data.runs.length} run${data.runs.length === 1 ? "" : "s"}`;
}

function visibleRuns() {
  const needle = state.filter.trim().toLowerCase();
  let runs = state.runs.filter((run) => {
    if (state.call !== "all" && run.direction !== state.call) return false;
    if (!needle) return true;
    return [run.ticker, run.action, run.rating, run.executive_summary, run.reasoning, run.run_id]
      .filter(Boolean)
      .some((field) => field.toLowerCase().includes(needle));
  });
  const by = {
    newest: (a, b) => b.sort_key.localeCompare(a.sort_key),
    oldest: (a, b) => a.sort_key.localeCompare(b.sort_key),
    ticker: (a, b) => a.ticker.localeCompare(b.ticker) || b.sort_key.localeCompare(a.sort_key),
    longest: (a, b) => b.words - a.words,
  };
  return runs.sort(by[state.sort] || by.newest);
}

/* ── tape ────────────────────────────────────────────────────────────── */

function renderStrip() {
  const runs = state.runs;
  const tickers = new Set(runs.map((run) => run.ticker));
  const total = runs.reduce((sum, run) => sum + run.words, 0);
  const counts = { bull: 0, bear: 0, flat: 0, none: 0 };
  runs.forEach((run) => { counts[run.direction] = (counts[run.direction] || 0) + 1; });
  const scored = counts.bull + counts.bear + counts.flat || 1;
  const newest = runs[0];

  $("#digest-strip").innerHTML = `
    <div class="stat">
      <div class="stat-key">Runs archived</div>
      <div class="stat-val">${runs.length}</div>
      <div class="stat-sub">${tickers.size} ticker${tickers.size === 1 ? "" : "s"}</div>
    </div>
    <div class="stat">
      <div class="stat-key">Latest</div>
      <div class="stat-val">${newest ? esc(newest.ticker) : "—"}</div>
      <div class="stat-sub">${newest ? esc(stampOf(newest)) : "no runs yet"}</div>
    </div>
    <div class="stat">
      <div class="stat-key">Words on file</div>
      <div class="stat-val">${words(total)}<small> w</small></div>
      <div class="stat-sub">${Math.round(total / 220) || 0} min of reading</div>
    </div>
    <div class="stat split">
      <div class="stat-key">Calls</div>
      <div class="split-bar">
        <i class="bull" style="width:${(counts.bull / scored) * 100}%"></i>
        <i class="flat" style="width:${(counts.flat / scored) * 100}%"></i>
        <i class="bear" style="width:${(counts.bear / scored) * 100}%"></i>
      </div>
      <div class="split-legend">
        <span class="bull"><b>${counts.bull}</b> buy</span>
        <span class="flat"><b>${counts.flat}</b> hold</span>
        <span class="bear"><b>${counts.bear}</b> sell</span>
      </div>
    </div>`;
}

function renderChips() {
  const counts = { all: state.runs.length };
  state.runs.forEach((run) => { counts[run.direction] = (counts[run.direction] || 0) + 1; });
  $("#call-chips").innerHTML = CALLS.map((call) => `
    <button class="chip" data-call="${call.key}" aria-pressed="${state.call === call.key}">
      ${call.label} ${counts[call.key] || 0}
    </button>`).join("");
}

function renderTape() {
  const runs = visibleRuns();
  const tape = $("#tape");
  tape.innerHTML = runs.map((run, index) => {
    const open = state.expandAll || state.expanded.has(run.run_id);
    const levels = levelCells(run) + horizonCell(run);
    return `
    <div class="run${open ? " open" : ""}" data-dir="${run.direction}" data-run="${esc(run.run_id)}"
         style="--i:${Math.min(index, 14)}">
      <div class="run-id">
        <a class="tick" href="#/run/${encodeURIComponent(run.run_id)}">${esc(run.ticker)}</a>
        <span class="stamp">${esc(stampOf(run))}</span>
      </div>
      <div class="run-meta">${run.sections.length} sections · ${words(run.words)} words · ${run.minutes} min</div>
      <div class="run-call">
        ${badge(run)}
        ${run.rating && run.rating !== run.action ? `<span class="rating">${esc(run.rating)}</span>` : ""}
      </div>
      <dl class="levels">${levels || `<div><dt>Levels</dt><dd class="thin">not quoted</dd></div>`}</dl>
      <div class="run-gist">${esc(run.executive_summary || run.reasoning || "No portfolio decision was written for this run.")}</div>
      <button class="chev" data-toggle="${esc(run.run_id)}" aria-label="Expand summary" title="Expand (e)">▸</button>
      <div class="run-more">${moreBlocks(run)}</div>
    </div>`;
  }).join("");

  const empty = $("#tape-empty");
  if (!runs.length) {
    empty.hidden = false;
    empty.innerHTML = state.runs.length
      ? "Nothing matches that filter."
      : `No reports under <code>${esc(state.root)}</code>.<br>Run <code>python analyze.py NVDA</code>, keep the “save report” answer, then hit Refresh.`;
  } else {
    empty.hidden = true;
  }
}

function moreBlocks(run) {
  const blocks = [
    ["Portfolio manager", run.executive_summary],
    ["Trader's case", run.reasoning],
    ["Position sizing", run.position_sizing],
    ["Sentiment", run.sentiment
      ? `${run.sentiment}${run.sentiment_score !== null ? ` · ${run.sentiment_score}/10` : ""}${run.sentiment_confidence ? ` · ${run.sentiment_confidence} confidence` : ""}`
      : null],
  ].filter(([, body]) => body);
  return `<div class="more-grid">
    ${blocks.map(([key, body]) => `<div class="more-block"><h4>${key}</h4><p>${esc(body)}</p></div>`).join("")}
    <div class="more-block more-open"><a class="btn btn-accent" href="#/run/${encodeURIComponent(run.run_id)}">Read in full →</a></div>
  </div>`;
}

/* ── digest sheet ────────────────────────────────────────────────────── */

function renderSheet() {
  $("#sheet").innerHTML = visibleRuns().map((run) => {
    const bits = [];
    if (run.entry_price !== null) bits.push(`entry <b>${num(run.entry_price)}</b>`);
    if (run.stop_loss !== null) bits.push(`stop <b>${num(run.stop_loss)}</b>`);
    if (run.price_target !== null) bits.push(`target <b>${num(run.price_target)}</b>`);
    if (run.time_horizon) bits.push(`<b>${esc(run.time_horizon)}</b>`);
    return `
      <article class="card" data-dir="${run.direction}">
        <div class="card-head">
          <span class="tick">${esc(run.ticker)}</span>
          ${badge(run)}
          <span class="stamp">${esc(stampOf(run))}</span>
        </div>
        <div class="card-levels">${bits.join(" · ") || "no levels quoted"}</div>
        <p>${esc(run.executive_summary || run.reasoning || "No decision written.")}</p>
        <a href="#/run/${encodeURIComponent(run.run_id)}">Read in full →</a>
      </article>`;
  }).join("") || `<p class="tape-empty">Nothing to digest yet.</p>`;
}

/* ── reader ──────────────────────────────────────────────────────────── */

async function openReader(runId) {
  showView("reader");
  $("#prose").innerHTML = `<p class="tape-empty">Loading ${esc(runId)}…</p>`;
  let payload;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    payload = await response.json();
  } catch (error) {
    $("#prose").innerHTML = `<p class="tape-empty">Could not open <code>${esc(runId)}</code>: ${esc(error.message)}</p>`;
    return;
  }
  state.run = payload;
  const run = payload.summary;
  document.title = `${run.ticker} · Report Room`;

  $("#reader-ticker").textContent = run.ticker;
  $("#reader-stamp").textContent = stampOf(run);
  $("#reader-call").innerHTML = badge(run) +
    (run.rating && run.rating !== run.action ? `<span class="rating">${esc(run.rating)}</span>` : "");
  $("#reader-levels").innerHTML = levelCells(run) + horizonCell(run) +
    `<div><dt>Length</dt><dd class="thin">${words(run.words)} w · ${run.minutes} min</dd></div>`;
  $("#btn-download").href = `/api/runs/${encodeURIComponent(run.run_id)}/markdown`;
  // Scoped to the reader so the tape's own per-row --call is never shadowed.
  $("#view-reader").style.setProperty("--call", `var(--${run.direction === "none" ? "fg-3" : run.direction})`);

  $("#prose").innerHTML = payload.sections.map((section) => `
    <section class="sec" id="sec-${section.id}" data-id="${section.id}">
      <div class="sec-head">
        <span class="sec-kicker">${section.numeral} · ${esc(section.group_label)}</span>
        <h3 class="sec-title">${esc(section.title)}</h3>
        <span class="sec-meta">${words(section.words)} words · ${Math.max(1, Math.round(section.words / 220))} min</span>
      </div>
      <div class="md">${section.html}</div>
    </section>`).join("") ||
    `<p class="tape-empty">This run has no section files on disk.</p>`;

  // Wide tables scroll inside their own box rather than stretching the column.
  $$(".md table", $("#prose")).forEach((table) => {
    const box = document.createElement("div");
    box.className = "table-scroll";
    table.replaceWith(box);
    box.appendChild(table);
  });

  renderRail();
  $("#find").value = "";
  state.matches = [];
  state.matchAt = -1;
  updateFindCount();
  window.scrollTo(0, 0);
  syncScroll();
}

function renderRail() {
  let lastGroup = null;
  $("#rail").innerHTML = state.run.sections.map((section) => {
    const head = section.group === lastGroup
      ? ""
      : `<div class="rail-group">${section.numeral} · ${esc(section.group_label)}</div>`;
    lastGroup = section.group;
    return `${head}<a class="rail-link" href="#sec-${section.id}" data-jump="${section.id}">
      ${esc(section.title)}<span>${words(section.words)}w</span></a>`;
  }).join("");
}

function renderOutline(sectionId) {
  const section = state.run?.sections.find((item) => item.id === sectionId);
  const outline = $("#outline");
  if (!section || !section.toc.length) { outline.innerHTML = ""; return; }
  outline.innerHTML = `<div class="outline-key">In this section</div>` +
    section.toc.map((item) => `<a class="lvl${item.level}" href="#${item.id}">${esc(item.title)}</a>`).join("");
}

/* One rAF-throttled scroll pass drives the progress bar, the rail highlight,
   and the outline highlight. */
let scrollQueued = false;
let activeSection = null;

function onScroll() {
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => { scrollQueued = false; syncScroll(); });
}

function syncScroll() {
  if ($("#view-reader").hidden) return;
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  $("#progress-bar").style.width = `${scrollable > 0 ? Math.min(100, (window.scrollY / scrollable) * 100) : 0}%`;

  const sections = $$(".sec", $("#prose"));
  let current = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= 130) current = section;
  }
  if (!current) return;
  if (current.dataset.id !== activeSection) {
    activeSection = current.dataset.id;
    $$(".rail-link").forEach((link) => link.classList.toggle("active", link.dataset.jump === activeSection));
    renderOutline(activeSection);
  }
  const headings = $$("h2[id], h3[id]", current);
  let heading = null;
  for (const item of headings) {
    if (item.getBoundingClientRect().top <= 140) heading = item;
  }
  $$("#outline a").forEach((link) => link.classList.toggle("active", heading && link.hash === `#${heading.id}`));
}

/* ── find in report ──────────────────────────────────────────────────── */

function clearMarks() {
  const prose = $("#prose");
  $$("mark", prose).forEach((mark) => mark.replaceWith(document.createTextNode(mark.textContent)));
  prose.normalize();
}

function applyFind(query) {
  clearMarks();
  if (!state.run) return;
  state.matches = [];
  state.matchAt = -1;
  const needle = query.trim().toLowerCase();
  if (needle.length >= 2) {
    const prose = $("#prose");
    const walker = document.createTreeWalker(prose, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) =>
        node.nodeValue.toLowerCase().includes(needle) && node.parentNode.nodeName !== "MARK"
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT,
    });
    const targets = [];
    while (walker.nextNode()) targets.push(walker.currentNode);
    targets.forEach((node) => {
      const text = node.nodeValue;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      for (;;) {
        const at = text.toLowerCase().indexOf(needle, cursor);
        if (at === -1) break;
        if (at > cursor) fragment.appendChild(document.createTextNode(text.slice(cursor, at)));
        const mark = document.createElement("mark");
        mark.textContent = text.slice(at, at + needle.length);
        fragment.appendChild(mark);
        cursor = at + needle.length;
      }
      if (cursor < text.length) fragment.appendChild(document.createTextNode(text.slice(cursor)));
      node.replaceWith(fragment);
    });
    state.matches = $$("mark", prose);
  }
  $$(".sec", $("#prose")).forEach((section) => {
    const hits = section.querySelectorAll("mark").length;
    const link = $(`.rail-link[data-jump="${section.dataset.id}"]`);
    if (!link) return;
    const meta = state.run.sections.find((item) => item.id === section.dataset.id);
    link.classList.toggle("hit", hits > 0);
    link.querySelector("span").textContent = hits ? `${hits} hit${hits === 1 ? "" : "s"}` : `${words(meta.words)}w`;
  });
  updateFindCount();
}

function updateFindCount() {
  const query = $("#find").value.trim();
  const count = state.matches.length;
  $("#find-count").textContent = !query || query.length < 2
    ? ""
    : count
      ? `${state.matchAt + 1 || 1}/${count}`
      : "none";
}

function jumpMatch(step) {
  if (!state.matches.length) return;
  state.matches.forEach((mark) => mark.classList.remove("on"));
  state.matchAt = (state.matchAt + step + state.matches.length) % state.matches.length;
  const mark = state.matches[state.matchAt];
  mark.classList.add("on");
  mark.scrollIntoView({ block: "center", behavior: "smooth" });
  updateFindCount();
}

/* ── routing ─────────────────────────────────────────────────────────── */

function showView(name) {
  const wasTape = !$("#view-tape").hidden;
  if (wasTape && name !== "tape") state.tapeScroll = window.scrollY;
  $("#view-tape").hidden = name !== "tape";
  $("#view-digest").hidden = name !== "digest";
  $("#view-reader").hidden = name !== "reader";
  if (name !== "reader") {
    state.run = null;
    activeSection = null;
    document.title = "Report Room · TradingAgents";
  }
}

function route() {
  const raw = location.hash;
  if (raw && raw !== "#" && !raw.startsWith("#/")) return;  // in-page anchor
  const hash = raw.replace(/^#\/?/, "");
  if (hash.startsWith("run/")) {
    const runId = decodeURIComponent(hash.slice(4));
    if (state.run?.summary.run_id === runId) return;  // already open
    openReader(runId);
  } else if (hash === "digest") {
    showView("digest");
    renderSheet();
    window.scrollTo(0, 0);
  } else {
    showView("tape");
    window.scrollTo(0, state.tapeScroll);
  }
}

/* ── wiring ──────────────────────────────────────────────────────────── */

function init() {
  let theme = "ink";
  try { theme = localStorage.getItem(THEME_KEY) || "ink"; } catch { /* private mode */ }
  setTheme(theme);

  $("#btn-theme").addEventListener("click", () =>
    setTheme(document.documentElement.dataset.theme === "ink" ? "paper" : "ink"));
  $("#btn-refresh").addEventListener("click", async () => {
    await loadRuns();
    toast(`${state.runs.length} run${state.runs.length === 1 ? "" : "s"} on file`);
  });
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-print]")) window.print();
  });

  $("#filter").addEventListener("input", (event) => {
    state.filter = event.target.value;
    renderTape();
    renderSheet();
  });
  $("#sort").addEventListener("change", (event) => {
    state.sort = event.target.value;
    renderTape();
    renderSheet();
  });
  $("#call-chips").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    state.call = chip.dataset.call;
    renderChips();
    renderTape();
    renderSheet();
  });
  $("#btn-expand").addEventListener("click", toggleExpandAll);

  $("#tape").addEventListener("click", (event) => {
    const chevron = event.target.closest("[data-toggle]");
    if (chevron) {
      const id = chevron.dataset.toggle;
      if (state.expanded.has(id)) state.expanded.delete(id); else state.expanded.add(id);
      chevron.closest(".run").classList.toggle("open");
      return;
    }
    if (event.target.closest("a")) return;
    const row = event.target.closest(".run");
    if (row) location.hash = `#/run/${encodeURIComponent(row.dataset.run)}`;
  });
  $("#find").addEventListener("input", (event) => applyFind(event.target.value));
  $("#find").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      jumpMatch(event.shiftKey ? -1 : 1);
    }
  });
  $("#btn-copy").addEventListener("click", copyMarkdown);

  window.addEventListener("hashchange", route);
  window.addEventListener("scroll", onScroll, { passive: true });
  document.addEventListener("keydown", onKey);
  // A run that finishes while this tab sits in the background shows up when the
  // user comes back to it.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && $("#view-reader").hidden && Date.now() - state.lastFetch > 5000) {
      loadRuns({ quiet: true });
    }
  });

  loadRuns().then(route).catch((error) => {
    $("#tape-empty").hidden = false;
    $("#tape-empty").textContent = `Could not reach the report API: ${error.message}`;
  });
}

function toggleExpandAll() {
  state.expandAll = !state.expandAll;
  state.expanded.clear();
  $("#btn-expand").textContent = state.expandAll ? "Collapse all" : "Expand all";
  renderTape();
}

async function copyMarkdown() {
  if (!state.run) return;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(state.run.summary.run_id)}/markdown`);
    const text = await response.text();
    await navigator.clipboard.writeText(text);
    toast(`Copied ${words(text.split(/\s+/).length)} words`);
  } catch {
    toast("Copy blocked — use Download");
  }
}

function onKey(event) {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.nodeName);
  if (event.key === "Escape") {
    if (typing && $("#find").value) {
      $("#find").value = "";
      applyFind("");
      $("#find").blur();
    } else if (!$("#view-reader").hidden || !$("#view-digest").hidden) {
      location.hash = "#/";
    }
    return;
  }
  if (typing || event.metaKey || event.ctrlKey || event.altKey) return;

  if (event.key === "/") {
    event.preventDefault();
    ($("#view-reader").hidden ? $("#filter") : $("#find")).focus();
    return;
  }
  if (event.key === "t") { setTheme(document.documentElement.dataset.theme === "ink" ? "paper" : "ink"); return; }
  if (event.key === "e" && !$("#view-tape").hidden) { toggleExpandAll(); return; }
  if ((event.key === "j" || event.key === "k") && !$("#view-reader").hidden && state.run) {
    const ids = state.run.sections.map((section) => section.id);
    const at = ids.indexOf(activeSection);
    const next = ids[Math.min(ids.length - 1, Math.max(0, (at === -1 ? 0 : at) + (event.key === "j" ? 1 : -1)))];
    if (next) $(`#sec-${next}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  }
}

init();
