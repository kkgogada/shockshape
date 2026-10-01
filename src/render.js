// Canvas rendering: flow view (airfoil, handles, wave system / Mach field),
// surface-pressure plot and optimizer history.
import { obliqueShock } from './gasdynamics.js';

export const COLORS = {
  red: '#e5383b', steel: '#9fb3c8', text: '#e7e9ec', muted: '#8d949e', faint: '#4a5058',
  grid: '#1f2329', bg: '#0e1013', airfoil: '#1d2026',
};

// ---------- colour maps ----------
function lerpColor(stops, v) {
  if (v <= stops[0][0]) return stops[0][1];
  for (let k = 1; k < stops.length; k++) {
    if (v <= stops[k][0]) {
      const [v0, c0] = stops[k - 1], [v1, c1] = stops[k];
      const t = (v - v0) / (v1 - v0);
      return c0.map((c, i) => Math.round(c + t * (c1[i] - c)));
    }
  }
  return stops[stops.length - 1][1];
}
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
export const MACH_STOPS = [
  [0.4, hex('#0f141b')], [0.75, hex('#2b4560')], [0.95, hex('#8aa5c0')], [1.0, hex('#e8eaed')],
  [1.06, hex('#f08a8b')], [1.25, hex('#e5383b')], [1.6, hex('#5c0c0f')],
];
// pressure coefficient: expansion (blue-grey) <- 0 -> compression (red)
export const CP_STOPS = [
  [-0.3, hex('#6f93b8')], [-0.12, hex('#30465d')], [0, hex('#15181d')], [0.12, hex('#6b1b1d')], [0.3, hex('#e5383b')],
];
export const machColor = (m) => `rgb(${lerpColor(MACH_STOPS, m).join(',')})`;
const lerpRgb = (m) => lerpColor(MACH_STOPS, m);
export const cpColor = (cp, scale = 1) => `rgb(${lerpColor(CP_STOPS, cp / scale).join(',')})`;
export function gradientCss(stops, lo, hi) {
  const parts = stops.map(([v, c]) => `rgb(${c.join(',')}) ${(((v - lo) / (hi - lo)) * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

// ---------- canvas helpers ----------
export function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: r.width, H: r.height };
}

/** World <-> screen transform for the flow view (true aspect ratio: wave angles are physical). */
export function flowView(W, H) {
  const [x0, x1] = W < 600 ? [-0.12, 1.18] : [-0.32, 1.5]; // zoom in on narrow screens
  const s = W / (x1 - x0);
  const yc = 0;
  return {
    s, x0, x1,
    y0: yc - H / (2 * s), y1: yc + H / (2 * s),
    X: (x) => (x - x0) * s,
    Y: (y) => H / 2 - (y - yc) * s,
    inv: (px, py) => [px / s + x0, (H / 2 - py) / s + yc],
  };
}

// ---------- flow view ----------
export function drawFlow(canvas, state) {
  const { ctx, W, H } = fitCanvas(canvas);
  const v = flowView(W, H);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, H);
  const r = state.result;
  const mode = r && r.valid ? (r.field ? 'mach' : r.waves ? 'pressure' : 'none') : 'none';
  if (mode === 'mach') drawMachField(ctx, v, r, W, H);
  if (mode === 'pressure') drawWaveSystem(ctx, v, r, W, H, state.geometry);
  drawAxes(ctx, v, W, H);
  drawAirfoil(ctx, v, state.geometry, state.handles, state.activeHandle, state.locked);
  drawFreestream(ctx, v, state.M, state.alpha);
  if (state.ghost) drawGhost(ctx, v, state.ghost);
  return { mode, view: v };
}

function drawAxes(ctx, v, W, H) {
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = -0.25; x <= 1.5; x += 0.25) { ctx.moveTo(v.X(x), 0); ctx.lineTo(v.X(x), H); }
  ctx.stroke();
  ctx.fillStyle = COLORS.faint;
  ctx.font = '11px JetBrains Mono, monospace';
  for (const x of [0, 0.5, 1]) ctx.fillText(`x/c ${x}`, v.X(x) + 4, H - 8);
}

function drawAirfoil(ctx, v, g, handles, active, locked) {
  if (!g) return;
  ctx.beginPath();
  ctx.moveTo(v.X(g.x[0]), v.Y(g.yu[0]));
  for (let k = 1; k < g.x.length; k++) ctx.lineTo(v.X(g.x[k]), v.Y(g.yu[k]));
  for (let k = g.x.length - 1; k >= 0; k--) ctx.lineTo(v.X(g.x[k]), v.Y(g.yl[k]));
  ctx.closePath();
  ctx.fillStyle = COLORS.airfoil;
  ctx.fill();
  ctx.strokeStyle = '#d7dbe0';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  if (!handles || locked) return;
  handles.forEach((h, k) => {
    const px = v.X(h.x), py = v.Y(h.y);
    ctx.beginPath();
    ctx.arc(px, py, k === active ? 7 : 5.5, 0, 2 * Math.PI);
    ctx.fillStyle = k === active ? '#fff' : COLORS.red;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = COLORS.bg;
    ctx.stroke();
  });
}

function drawGhost(ctx, v, g) {
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(231,233,236,0.45)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(v.X(g.x[0]), v.Y(g.yu[0]));
  for (let k = 1; k < g.x.length; k++) ctx.lineTo(v.X(g.x[k]), v.Y(g.yu[k]));
  for (let k = g.x.length - 1; k >= 0; k--) ctx.lineTo(v.X(g.x[k]), v.Y(g.yl[k]));
  ctx.stroke();
  ctx.restore();
}

function drawFreestream(ctx, v, M, alphaDeg) {
  const a = (alphaDeg * Math.PI) / 180;
  const x = 18, y = 28, L = 46;
  ctx.save();
  ctx.strokeStyle = COLORS.text;
  ctx.fillStyle = COLORS.text;
  ctx.lineWidth = 1.5;
  const ex = x + L * Math.cos(a), ey = y - L * Math.sin(a);
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
  const ah = 7;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - ah * Math.cos(a - 0.4), ey + ah * Math.sin(a - 0.4));
  ctx.lineTo(ex - ah * Math.cos(a + 0.4), ey + ah * Math.sin(a + 0.4));
  ctx.closePath(); ctx.fill();
  ctx.font = '12px JetBrains Mono, monospace';
  ctx.fillText(`M∞ ${M.toFixed(2)}   α ${alphaDeg.toFixed(1)}°`, x + L + 10, y + 4);
  ctx.restore();
}

/** Local Mach number heat map from the TSD solution, with the sonic line and shocks. */
function drawMachField(ctx, v, r, W, H) {
  const { x, y, NI, NJ, mach } = r.field;
  // Bilinear interpolation of the nodal Mach field onto a coarse pixel
  // lattice, then scaled up smoothly: avoids the blocky look of the
  // stretched grid while showing exactly the computed values at nodes.
  const step = 3;
  const nx = Math.ceil(W / step), ny = Math.ceil(H / step);
  const img = new ImageData(nx, ny);
  const bracket = (arr, val) => {
    let lo = 0, hi = arr.length - 1;
    if (val <= arr[lo]) return [0, 0];
    if (val >= arr[hi]) return [hi - 1, 1];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (arr[m] <= val) lo = m; else hi = m; }
    return [lo, (val - arr[lo]) / (arr[lo + 1] - arr[lo])];
  };
  const jCut = r.field.NJ / 2; // rows below the cut are j < jCut
  const xi = [], yj = [];
  for (let px = 0; px < nx; px++) xi.push(bracket(x, v.inv((px + 0.5) * step, 0)[0]));
  for (let py = 0; py < ny; py++) {
    const yy = v.inv(0, (py + 0.5) * step)[1];
    let [j, t] = bracket(y, yy);
    // never interpolate across the wake cut / airfoil: snap to the nearest row on the same side
    if (j === jCut - 1) { if (yy >= 0) { j = jCut; t = 0; } else { t = 1; } }
    yj.push([j, t]);
  }
  for (let py = 0; py < ny; py++) {
    const [j, ty] = yj[py];
    for (let px = 0; px < nx; px++) {
      const [i, tx] = xi[px];
      const i1 = Math.min(i + 1, NI - 1), j1 = Math.min(j + 1, NJ - 1);
      const m = (1 - tx) * ((1 - ty) * mach[i * NJ + j] + ty * mach[i * NJ + j1]) +
        tx * ((1 - ty) * mach[i1 * NJ + j] + ty * mach[i1 * NJ + j1]);
      const c = lerpRgb(m);
      const o = 4 * (py * nx + px);
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(nx, ny);
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, 0, 0, nx * step, ny * step);
  ctx.restore();
  // sonic line: marching squares on M = 1
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  for (let i = 1; i < NI - 2; i++) {
    if (x[i + 1] < v.x0 || x[i] > v.x1) continue;
    for (let j = 0; j < NJ - 1; j++) {
      if (y[j + 1] < v.y0 || y[j] > v.y1) continue;
      if (j === r.field.NJ / 2 - 1) continue; // don't connect across the wake cut
      const c = [mach[i * NJ + j], mach[(i + 1) * NJ + j], mach[(i + 1) * NJ + j + 1], mach[i * NJ + j + 1]];
      const px = [x[i], x[i + 1], x[i + 1], x[i]], py = [y[j], y[j], y[j + 1], y[j + 1]];
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const a = c[e] - 1, b = c[(e + 1) % 4] - 1;
        if ((a < 0) !== (b < 0)) {
          const t = a / (a - b);
          pts.push([px[e] + t * (px[(e + 1) % 4] - px[e]), py[e] + t * (py[(e + 1) % 4] - py[e])]);
        }
      }
      if (pts.length >= 2) {
        ctx.moveTo(v.X(pts[0][0]), v.Y(pts[0][1]));
        ctx.lineTo(v.X(pts[1][0]), v.Y(pts[1][1]));
      }
    }
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // shocks: join the detected shock points on each side into a line
  if (r.shocks?.length) {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.2;
    ctx.lineJoin = 'round';
    for (const side of [1, -1]) {
      const pts = r.shocks.filter((p) => p.jump > 0.008 && Math.sign(p.y) === side).sort((a, b) => Math.abs(a.y) - Math.abs(b.y));
      ctx.beginPath();
      let prev = null;
      for (const p of pts) {
        if (prev && Math.abs(p.y - prev.y) < 0.12 && Math.abs(p.x - prev.x) < 0.08) ctx.lineTo(v.X(p.x), v.Y(p.y));
        else ctx.moveTo(v.X(p.x), v.Y(p.y));
        prev = p;
      }
      ctx.stroke();
    }
  }
}

/** Shock-expansion wave system: LE shocks / fans, simple-wave regions, approximate TE waves. */
function drawWaveSystem(ctx, v, r, W, H, g) {
  const w = r.waves;
  const M = w.Minf, a = w.alpha;
  const scale = Math.max(0.15, Math.min(0.5, 1.6 / (M * M)));
  ctx.fillStyle = cpColor(0, scale);
  ctx.fillRect(0, 0, W, H);
  const reach = 3;
  for (const side of ['upper', 'lower']) {
    const s = w[side];
    const sign = side === 'upper' ? 1 : -1;
    const ys = side === 'upper' ? g.yu : g.yl;
    const cps = side === 'upper' ? r.cpU : r.cpL;
    // leading-edge wave
    let leRay; // boundary of the disturbed region
    if (s.le.type === 'shock') {
      leRay = a + sign * s.le.beta;
    } else {
      leRay = a + sign * Math.asin(1 / M);
    }
    const leDir = [Math.cos(leRay), Math.sin(leRay)];
    // Mach lines from the surface, each carrying its surface Cp (simple wave)
    const n = s.mach.length;
    const lines = [];
    for (let k = 0; k <= n; k++) {
      const kk = Math.min(k, n - 1);
      const mu = Math.asin(1 / s.mach[kk]);
      const ang = s.ang[kk] + sign * mu;
      const px = g.x[k], py = ys[k];
      const d = [Math.cos(ang), Math.sin(ang)];
      // intersect with the LE wave ray from (0,0)
      let len = reach;
      const det = d[0] * leDir[1] - d[1] * leDir[0];
      if (Math.abs(det) > 1e-9) {
        const t = (px * leDir[1] - py * leDir[0]) / det; // param along d (negative sign convention)
        const tt = -t;
        if (tt >= -1e-12 && tt < len) len = Math.max(0, tt);
      }
      lines.push({ p: [px, py], q: [px + len * d[0], py + len * d[1]], cp: cps[kk] });
    }
    // centred expansion fan at the leading edge: fill with interpolated slices
    if (s.le.type === 'expansion') {
      const a0 = leRay, a1 = Math.atan2(lines[0].q[1], lines[0].q[0]);
      const slices = 10;
      for (let q = 0; q < slices; q++) {
        const t0 = q / slices, t1 = (q + 1) / slices;
        const b0 = a0 + (a1 - a0) * t0, b1 = a0 + (a1 - a0) * t1;
        ctx.fillStyle = cpColor(cps[0] * (t0 + t1) / 2, scale);
        ctx.beginPath();
        ctx.moveTo(v.X(0), v.Y(0));
        ctx.lineTo(v.X(reach * Math.cos(b0)), v.Y(reach * Math.sin(b0)));
        ctx.lineTo(v.X(reach * Math.cos(b1)), v.Y(reach * Math.sin(b1)));
        ctx.closePath();
        ctx.fill();
      }
    }
    // fill quads between neighbouring Mach lines
    for (let k = 0; k < lines.length - 1; k++) {
      const A = lines[k], B = lines[k + 1];
      ctx.fillStyle = cpColor(A.cp, scale);
      ctx.beginPath();
      ctx.moveTo(v.X(A.p[0]), v.Y(A.p[1]));
      ctx.lineTo(v.X(A.q[0]), v.Y(A.q[1]));
      ctx.lineTo(v.X(B.q[0]), v.Y(B.q[1]));
      ctx.lineTo(v.X(B.p[0]), v.Y(B.p[1]));
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
    // a few Mach lines for texture
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 4; k < lines.length; k += 8) {
      ctx.moveTo(v.X(lines[k].p[0]), v.Y(lines[k].p[1]));
      ctx.lineTo(v.X(lines[k].q[0]), v.Y(lines[k].q[1]));
    }
    ctx.stroke();
    // LE wave
    ctx.strokeStyle = s.le.type === 'shock' ? '#ffffff' : 'rgba(159,179,200,0.8)';
    ctx.lineWidth = s.le.type === 'shock' ? 2 : 1;
    ctx.setLineDash(s.le.type === 'shock' ? [] : [4, 3]);
    ctx.beginPath();
    ctx.moveTo(v.X(0), v.Y(0));
    ctx.lineTo(v.X(reach * leDir[0]), v.Y(reach * leDir[1]));
    ctx.stroke();
    ctx.setLineDash([]);
    // trailing-edge wave (approximate: turn each stream back to the free-stream direction)
    const Mte = s.mach[n - 1];
    const turnBack = sign > 0 ? a - s.ang[n - 1] : s.ang[n - 1] - a; // compression if > 0
    let teAng;
    if (turnBack > 0) {
      const sh = obliqueShock(Mte, turnBack);
      teAng = sh ? s.ang[n - 1] + sign * sh.beta : null;
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
    } else {
      teAng = a + sign * Math.asin(1 / Math.max(M, 1.0001));
      ctx.strokeStyle = 'rgba(159,179,200,0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
    }
    if (teAng !== null) {
      ctx.beginPath();
      ctx.moveTo(v.X(1), v.Y(ys[ys.length - 1]));
      ctx.lineTo(v.X(1 + reach * Math.cos(teAng)), v.Y(ys[ys.length - 1] + reach * Math.sin(teAng)));
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
}

// ---------- Cp plot ----------
export function drawCp(canvas, r, opts = {}) {
  const { ctx, W, H } = fitCanvas(canvas);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, H);
  const pad = { l: 46, r: 14, t: 12, b: 26 };
  if (!r || !r.valid) {
    ctx.fillStyle = COLORS.muted;
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('No valid solution for this case.', pad.l, H / 2);
    return;
  }
  const series = [
    { x: r.x, y: r.cpU, color: COLORS.red, w: 2 },
    { x: r.x, y: r.cpL, color: COLORS.steel, w: 2 },
  ];
  if (r.linear) {
    series.push({ x: r.x, y: r.linear.cpU, color: 'rgba(229,56,59,0.6)', w: 1.2, dash: [5, 4] });
    series.push({ x: r.x, y: r.linear.cpL, color: 'rgba(159,179,200,0.6)', w: 1.2, dash: [5, 4] });
  }
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (let k = 0; k < s.y.length; k++) {
    if (s.x[k] < 0.01 || s.x[k] > 0.99) continue; // keep LE/TE singular points from setting the scale
    lo = Math.min(lo, -s.y[k]); hi = Math.max(hi, -s.y[k]);
  }
  if (opts.cpStar !== undefined) { lo = Math.min(lo, -opts.cpStar); hi = Math.max(hi, -opts.cpStar); }
  const span = Math.max(hi - lo, 0.1);
  lo -= 0.08 * span; hi += 0.08 * span;
  const X = (x) => pad.l + x * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  // grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillStyle = COLORS.faint;
  const step = niceStep(hi - lo);
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(t)); ctx.lineTo(W - pad.r, Y(t)); ctx.stroke();
    ctx.fillText((-t).toFixed(2).replace('-0.00', '0.00'), 4, Y(t) + 4);
  }
  for (const x of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(String(x), X(x) - 6, H - 8);
  ctx.strokeStyle = '#3a3f47';
  ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(W - pad.r, Y(0)); ctx.stroke();
  ctx.fillStyle = COLORS.muted;
  ctx.fillText('Cp (negative up)', pad.l + 6, pad.t + 10);
  ctx.fillText('↑ suction', W - pad.r - 66, pad.t + 10);
  if (opts.cpStar !== undefined) {
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = '#e7e9ec';
    ctx.beginPath(); ctx.moveTo(pad.l, Y(-opts.cpStar)); ctx.lineTo(W - pad.r, Y(-opts.cpStar)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.text;
    ctx.fillText('Cp* (sonic)', pad.l + 6, Y(-opts.cpStar) - 5);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b); ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.w;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    for (let k = 0; k < s.x.length; k++) {
      const px = X(s.x[k]), py = Y(-s.y[k]);
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
  ctx.setLineDash([]);
}

function niceStep(span) {
  const raw = span / 5;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

// ---------- optimizer history ----------
export function drawHistory(canvas, hist) {
  const { ctx, W, H } = fitCanvas(canvas);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, H);
  const pad = { l: 64, r: 64, t: 10, b: 20 };
  ctx.font = '11px JetBrains Mono, monospace';
  if (!hist.length) {
    ctx.fillStyle = COLORS.faint;
    ctx.fillText('Convergence history appears here.', pad.l, H / 2 + 4);
    return;
  }
  const n = Math.max(hist.length - 1, 1);
  const cds = hist.map((h) => h.cd), cls = hist.map((h) => h.cl);
  const rng = (a) => { let lo = Math.min(...a), hi = Math.max(...a); if (hi - lo < 1e-9) { lo -= 1e-3; hi += 1e-3; } return [lo, hi]; };
  const [c0, c1] = rng(cds), [l0, l1] = rng(cls);
  const X = (k) => pad.l + (k / n) * (W - pad.l - pad.r);
  const Yd = (v) => pad.t + (1 - (v - c0) / (c1 - c0)) * (H - pad.t - pad.b);
  const Yl = (v) => pad.t + (1 - (v - l0) / (l1 - l0)) * (H - pad.t - pad.b);
  ctx.strokeStyle = COLORS.grid;
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  const line = (ys, Y, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    ys.forEach((v, k) => (k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v))));
    ctx.stroke();
  };
  line(cls, Yl, COLORS.steel);
  line(cds, Yd, COLORS.red);
  ctx.fillStyle = COLORS.red;
  ctx.fillText(`CD ${c1.toFixed(4)}`, 4, pad.t + 9);
  ctx.fillText(`   ${c0.toFixed(4)}`, 4, H - pad.b);
  ctx.fillStyle = COLORS.steel;
  ctx.fillText(`CL ${l1.toFixed(3)}`, W - pad.r + 6, pad.t + 9);
  ctx.fillText(`   ${l0.toFixed(3)}`, W - pad.r + 6, H - pad.b);
  ctx.fillStyle = COLORS.faint;
  ctx.fillText(`iteration ${hist[hist.length - 1].iter}`, X(n) - 90, H - 5);
}
