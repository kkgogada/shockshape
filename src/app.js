import { HANDLE_X, N_HANDLES, PRESETS, surfaces, cosineStations, area, maxThickness } from './geometry.js';
import { cpCritical } from './gasdynamics.js';
import { ENGINE_INFO, SUPERSONIC_SWITCH } from './engines.js';
import { drawFlow, drawCp, drawHistory, flowView, gradientCss, MACH_STOPS, CP_STOPS, COLORS } from './render.js';

const $ = (id) => document.getElementById(id);
const XS = cosineStations(161);

const state = {
  shape: PRESETS['Biconvex 6%'].slice(),
  M: 2.0,
  alpha: 2,
  engine: 'auto',
  result: null,
  geometry: null,
  activeHandle: -1,
  locked: false,
  ghost: null,
  history: [],
  startDesign: null,
  surrogate: null,
};

// ---------------- worker plumbing ----------------
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let reqId = 0, inflight = false, queued = false, workerReady = false, optId = null;

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'ready') {
    workerReady = true;
    state.surrogate = m.surrogate;
    if (!m.surrogate) $('engine').querySelector('[value=surrogate]').disabled = true;
    requestSolve();
  } else if (m.type === 'result') {
    inflight = false;
    if (m.id === reqId) { state.result = m.result; state.solveMs = m.ms; }
    showBusy(false);
    render();
    if (queued) { queued = false; requestSolve(); }
  } else if (m.type === 'opt-progress' && m.id === optId) {
    state.shape = m.rec.shape.slice();
    state.alpha = m.rec.alpha;
    $('alpha').value = state.alpha;
    state.result = m.result;
    state.history.push(m.rec);
    state.optEvals = m.rec.evals;
    state.optMs = m.ms;
    render();
    updateOptText(false);
  } else if (m.type === 'opt-baseline' && m.id === optId) {
    state.baseline = m.baseline;
  } else if (m.type === 'opt-done' && m.id === optId) {
    finishOptimization(m);
  } else if (m.type === 'opt-error' && m.id === optId) {
    finishOptimization({ error: m.message });
  }
};

function requestSolve() {
  if (!workerReady || state.locked) return;
  if (inflight) { queued = true; return; }
  inflight = true;
  reqId++;
  const which = resolvedEngine();
  if (which === 'tsd') showBusy(true, 'Solving transonic flow…');
  worker.postMessage({ type: 'evaluate', id: reqId, engine: state.engine, shape: state.shape, M: state.M, alpha: state.alpha });
}

let busyTimer = null;
function showBusy(on, text) {
  clearTimeout(busyTimer);
  if (on) {
    busyTimer = setTimeout(() => { $('busy').hidden = false; $('busyText').textContent = text; }, 120);
  } else {
    $('busy').hidden = true;
  }
}

function resolvedEngine() {
  if (state.engine !== 'auto') return state.engine;
  return state.M >= SUPERSONIC_SWITCH ? 'shock-expansion' : 'tsd';
}

// ---------------- rendering ----------------
function handles() {
  const hs = [];
  for (let j = 0; j < N_HANDLES; j++) hs.push({ x: HANDLE_X[j], y: state.shape[j], k: j });
  for (let j = 0; j < N_HANDLES; j++) hs.push({ x: HANDLE_X[j], y: state.shape[N_HANDLES + j], k: N_HANDLES + j });
  return hs;
}

function render() {
  state.geometry = surfaces(state.shape, XS);
  const { mode } = drawFlow($('flow'), { ...state, handles: handles() });
  const r = state.result;
  const cpStar = state.M < 1 ? cpCritical(state.M) : undefined;
  drawCp($('cp'), r, { cpStar });
  drawHistory($('hist'), state.history);
  updateReadouts(r);
  updateLegends(mode, r);
}

function fmt(v, d = 4) { return Number.isFinite(v) ? v.toFixed(d) : '–'; }

function updateReadouts(r) {
  const ok = r && r.valid;
  $('roCl').textContent = ok ? fmt(r.cl, 4) : '–';
  $('roCd').textContent = ok ? fmt(r.cd, 5) : '–';
  $('roLD').textContent = ok && r.cd > 1e-6 ? fmt(r.cl / r.cd, 1) : ok ? '∞' : '–';
  $('roCm').textContent = ok ? fmt(r.cm, 4) : '–';
  const t = maxThickness(state.shape);
  $('roTc').textContent = `${(t.t * 100).toFixed(2)}%`;
  $('roArea').textContent = area(state.shape).toFixed(4);
  const parts = [];
  if (r) {
    parts.push(ENGINE_INFO[r.engine]?.label ?? r.engine);
    if (r.solver) parts.push(`${r.solver.converged ? 'converged' : 'NOT converged'} · ${r.solver.totalSweeps} sweeps · ${r.solver.coldStart ? 'cold' : 'warm'} start`);
    if (state.solveMs !== undefined && !state.locked) parts.push(`${state.solveMs < 10 ? state.solveMs.toFixed(1) : Math.round(state.solveMs)} ms`);
    if (r.engine === 'surrogate' && state.surrogate) parts.push(`trained on: ${state.surrogate.source}`);
  }
  $('status').textContent = parts.join(' · ');
  const warns = r?.warnings ?? [];
  const ul = $('warnings');
  ul.replaceChildren(...warns.map((w) => Object.assign(document.createElement('li'), { textContent: w })));
  if (r && r.supercritical) {
    const li = document.createElement('li');
    li.textContent = `Supercritical: local Mach reaches ${r.maxLocalMach.toFixed(2)}${r.shocks?.length ? ', with a shock' : ''}.`;
    li.style.color = '#e7e9ec'; li.style.borderColor = COLORS.red; li.style.background = 'rgba(229,56,59,0.08)';
    ul.prepend(li);
  }
}

function updateLegends(mode, r) {
  const L = $('legend');
  if (mode === 'mach') {
    L.innerHTML = `<span>local Mach 0.4<span class="bar" style="background:${gradientCss(MACH_STOPS, 0.4, 1.6)}"></span>1.6</span><span><i class="dash" style="color:#fff"></i>sonic line</span><span><i style="background:#fff"></i>shock</span>`;
  } else if (mode === 'pressure') {
    L.innerHTML = `<span>expansion<span class="bar" style="background:${gradientCss(CP_STOPS, -0.3, 0.3)}"></span>compression</span><span><i style="background:#fff"></i>shock</span><span><i class="dash" style="color:${COLORS.steel}"></i>expansion fan edge</span>`;
  } else {
    L.innerHTML = r?.engine === 'surrogate' ? '<span>The surrogate predicts surface quantities only.</span>' : '';
  }
  const lin = r?.linear ? `<span><i class="dash" style="color:${COLORS.muted}"></i>linear (Ackeret)</span>` : '';
  const cps = state.M < 1 ? `<span><i class="dash" style="color:#e7e9ec"></i>Cp*</span>` : '';
  $('cpLegend').innerHTML = `<span><i style="background:${COLORS.red}"></i>upper</span><span><i style="background:${COLORS.steel}"></i>lower</span>${lin}${cps}`;
}

function updateRegime() {
  const M = state.M;
  let txt;
  if (M < 0.75) txt = '<b>Subsonic.</b> Inviscid, shock-free flow has no drag at all (d’Alembert).';
  else if (M < 1) txt = '<b>Transonic.</b> Pockets of supersonic flow can end in shocks: drag rise.';
  else if (M < SUPERSONIC_SWITCH) txt = '<b>Low supersonic.</b> Bow shock likely detached; TSD handles it.';
  else txt = '<b>Supersonic.</b> Attached shocks and expansion fans; wave drag dominates.';
  $('regime').innerHTML = txt;
  $('machOut').textContent = M.toFixed(2);
  $('alphaOut').textContent = `${state.alpha.toFixed(1)}°`;
  const eng = resolvedEngine();
  $('engineNote').textContent = (state.engine === 'auto' ? `Auto → ${ENGINE_INFO[eng].label}. ` : '') + ENGINE_INFO[eng].blurb;
  const g = ENGINE_INFO[eng].gradient;
  $('gradPill').textContent = g === 'exact' ? 'gradient: back-propagation (1 pass)' : `gradient: finite differences (${g === 'central' ? 22 : 11} solves / step)`;
}

// ---------------- interaction: dragging handles ----------------
const canvas = $('flow');
function pointer(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top, r.width, r.height];
}
canvas.addEventListener('pointerdown', (e) => {
  if (state.locked) return;
  const [px, py, W, H] = pointer(e);
  const v = flowView(W, H);
  let best = -1, bd = 14;
  handles().forEach((h, k) => {
    const d = Math.hypot(v.X(h.x) - px, v.Y(h.y) - py);
    if (d < bd) { bd = d; best = k; }
  });
  if (best >= 0) {
    state.activeHandle = best;
    canvas.setPointerCapture(e.pointerId);
    render();
  }
});
canvas.addEventListener('pointermove', (e) => {
  const [px, py, W, H] = pointer(e);
  const v = flowView(W, H);
  if (state.activeHandle < 0) {
    const near = !state.locked && handles().some((h) => Math.hypot(v.X(h.x) - px, v.Y(h.y) - py) < 14);
    canvas.style.cursor = near ? 'ns-resize' : 'default';
    return;
  }
  const [, y] = v.inv(px, py);
  const k = state.activeHandle;
  // keep each surface on its own side of its partner handle
  const partner = k < N_HANDLES ? state.shape[k + N_HANDLES] : state.shape[k - N_HANDLES];
  let ny = Math.max(-0.15, Math.min(0.15, y));
  if (k < N_HANDLES) ny = Math.max(ny, partner + 0.002); else ny = Math.min(ny, partner - 0.002);
  state.shape[k] = ny;
  state.history = [];
  render();
  requestSolve();
});
const endDrag = () => { if (state.activeHandle >= 0) { state.activeHandle = -1; render(); } };
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

// ---------------- controls ----------------
const preset = $('preset');
for (const name of Object.keys(PRESETS)) preset.append(new Option(name, name));
preset.addEventListener('change', () => {
  state.shape = PRESETS[preset.value].slice();
  $('limit').value = $('constraint').value === 'area' ? area(state.shape).toFixed(4) : maxThickness(state.shape).t.toFixed(3);
  state.history = []; state.ghost = null;
  render(); requestSolve();
});
$('mach').addEventListener('input', (e) => { state.M = +e.target.value; updateRegime(); render(); requestSolve(); });
$('alpha').addEventListener('input', (e) => { state.alpha = +e.target.value; updateRegime(); render(); requestSolve(); });
$('engine').addEventListener('change', (e) => { state.engine = e.target.value; updateRegime(); requestSolve(); });
$('clTarget').addEventListener('input', (e) => { $('clOut').textContent = (+e.target.value).toFixed(2); });
$('constraint').addEventListener('change', (e) => {
  const lim = $('limit');
  if (e.target.value === 'area') { lim.value = area(state.shape).toFixed(4); lim.step = '0.002'; }
  else { lim.value = maxThickness(state.shape).t.toFixed(3); lim.step = '0.005'; }
});

// ---------------- optimizer ----------------
$('run').addEventListener('click', () => {
  const engine = resolvedEngine();
  if (engine === 'tsd' && state.M < 0.75) {
    $('optText').innerHTML = '<b>Nothing to minimise here:</b> below the critical Mach number the inviscid flow is shock-free and wave drag is exactly zero. Raise the Mach number into the transonic range.';
    return;
  }
  state.locked = true;
  state.startDesign = { shape: state.shape.slice(), alpha: state.alpha };
  state.ghost = surfaces(state.shape, XS);
  state.history = [];
  state.baseline = null;
  state.optEngine = engine;
  optId = Date.now();
  setOptButtons(true);
  ['mach', 'alpha', 'engine', 'preset', 'clTarget', 'constraint', 'limit'].forEach((id) => ($(id).disabled = true));
  worker.postMessage({
    type: 'optimize', id: optId, engine, M: state.M,
    shape: state.shape, alpha: state.alpha,
    clTarget: +$('clTarget').value, constraint: $('constraint').value, limit: +$('limit').value,
    minStepMs: 140,
  });
  updateOptText(false);
  render();
});
$('stop').addEventListener('click', () => worker.postMessage({ type: 'stop' }));
$('restore').addEventListener('click', () => {
  if (!state.startDesign) return;
  state.shape = state.startDesign.shape.slice();
  state.alpha = state.startDesign.alpha;
  $('alpha').value = state.alpha;
  state.ghost = null; state.history = [];
  $('restore').disabled = true;
  updateRegime(); render(); requestSolve();
});

function setOptButtons(running) {
  $('run').disabled = running;
  $('stop').disabled = !running;
  $('restore').disabled = running || !state.startDesign;
}

function finishOptimization(m) {
  state.locked = false;
  setOptButtons(false);
  ['mach', 'alpha', 'engine', 'preset', 'clTarget', 'constraint', 'limit'].forEach((id) => ($(id).disabled = false));
  if (m.error) {
    $('optText').innerHTML = `<b>Could not start:</b> ${m.error}`;
  } else {
    updateOptText(true, m);
  }
  updateRegime();
  render();
  requestSolve(); // re-evaluate on the display grid / with the linear overlay
}

function verificationText(v) {
  if (!v) return '';
  const name = ENGINE_INFO[v.engine]?.label ?? v.engine;
  if (!v.valid) return `<br><b>Check with ${name}:</b> the optimized design is outside that model's validity (e.g. detached shock), so the surrogate's claim cannot be confirmed here.`;
  const err = Math.abs(v.cd - (state.history.at(-1)?.cd ?? v.cd)) / v.cd * 100;
  const gain = v.baseCd ? ` True reduction vs. the trimmed start: ${((1 - v.cd / v.baseCd) * 100).toFixed(1)}%.` : '';
  return `<br><b>Checked with ${name}:</b> C<sub>D</sub> ${v.cd.toFixed(5)} at C<sub>L</sub> ${v.cl.toFixed(3)} (surrogate C<sub>D</sub> error ${err.toFixed(1)}%).${gain}`;
}

function updateOptText(done, m) {
  const h = state.history;
  if (!h.length) {
    $('optText').innerHTML = `Starting ${ENGINE_INFO[state.optEngine].label.toLowerCase()} optimization…`;
    return;
  }
  const last = h[h.length - 1];
  const first = state.baseline ?? h[0];
  const red = first.cd > 0 ? (1 - last.cd / first.cd) * 100 : 0;
  const limit = $('constraint').value === 'area' ? `area ${last.area.toFixed(4)}` : `t/c ${(last.tc * 100).toFixed(2)}%`;
  const evals = `${last.evals} flow evaluations`;
  const per = last.iter ? ` (${Math.round(last.evals / Math.max(1, last.iter))} per iteration)` : '';
  const time = state.optMs ? `, ${(state.optMs / 1000).toFixed(1)} s of compute` : '';
  const base = state.baseline ? `starting shape trimmed to the same lift: C<sub>D</sub> ${first.cd.toFixed(5)}` : `start C<sub>D</sub> ${first.cd.toFixed(5)}`;
  const cdTxt = `${base} → optimized <b>${last.cd.toFixed(5)}</b> (${red >= 0 ? '−' : '+'}${Math.abs(red).toFixed(1)}%)`;
  const status = done ? (m?.stopped ? 'Stopped' : 'Converged') : `Iteration ${last.iter}`;
  $('optText').innerHTML = `<b>${status}.</b> ${cdTxt} at C<sub>L</sub> ${last.cl.toFixed(3)}, ${limit}. ${evals}${per}${time}.` +
    (state.optEngine === 'tsd' ? ' Finite-difference gradients through a shock-capturing solver are slow and noisy: exactly the cost an adjoint or a surrogate removes.' : '') +
    (state.optEngine === 'surrogate' ? ' Each gradient is one backward pass through the network.' : '') +
    verificationText(m?.verification);
}

// ---------------- boot ----------------
$('clOut').textContent = (+$('clTarget').value).toFixed(2);
$('limit').value = maxThickness(state.shape).t.toFixed(3);
$('mach').value = state.M;
$('alpha').value = state.alpha;
updateRegime();
render();
window.addEventListener('resize', render);
if (document.fonts) document.fonts.ready.then(render);
