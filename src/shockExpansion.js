// Shock-expansion theory and linearized (Ackeret) theory for sharp-edged
// airfoils in supersonic flow.
//
// Shock-expansion: an oblique shock (or Prandtl-Meyer expansion) at the
// leading edge, then isentropic simple-wave turning along each surface.
// For a thin, sharp section with an attached shock this is the exact
// inviscid surface solution apart from weak reflected waves, and it is
// the standard reduced model for this regime.

import { surfaces, cosineStations } from './geometry.js';
import {
  GAMMA, prandtlMeyer, inversePrandtlMeyer, pOverP0, obliqueShock,
  maxDeflection, cpFromPressureRatio,
} from './gasdynamics.js';

const N_STATIONS = 161;
const XS = cosineStations(N_STATIONS);

/** Panel geometry for one surface, ordered LE -> TE. */
function panels(x, y) {
  const n = x.length - 1;
  const xm = new Float64Array(n), ang = new Float64Array(n), ds = new Float64Array(n);
  const tx = new Float64Array(n), ty = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const dx = x[k + 1] - x[k], dy = y[k + 1] - y[k];
    ds[k] = Math.hypot(dx, dy);
    tx[k] = dx / ds[k];
    ty[k] = dy / ds[k];
    ang[k] = Math.atan2(dy, dx);
    xm[k] = 0.5 * (x[k] + x[k + 1]);
  }
  return { n, xm, ang, ds, tx, ty };
}

/**
 * March one surface. `turn[k]` is the flow deflection (compression positive)
 * the k-th panel imposes relative to the free stream.
 */
function marchSurface(Minf, turn, tmaxInfo) {
  const n = turn.length;
  const cp = new Float64Array(n), mach = new Float64Array(n);
  const res = { cp, mach, ok: true, reason: '', le: null };
  const d0 = turn[0];
  let M, p0ratio; // p0 local / p0 freestream
  if (d0 > 0) {
    if (d0 > tmaxInfo.theta) {
      res.ok = false;
      res.reason = `leading-edge deflection ${(d0 * 180 / Math.PI).toFixed(1)}° exceeds the attached-shock limit ${(tmaxInfo.theta * 180 / Math.PI).toFixed(1)}° — shock detaches`;
      return res;
    }
    const sh = obliqueShock(Minf, d0);
    M = sh.M2;
    p0ratio = sh.p02p01;
    res.le = { type: 'shock', beta: sh.beta };
  } else {
    const nu = prandtlMeyer(Minf) - d0; // d0 < 0 => expansion
    M = inversePrandtlMeyer(nu);
    p0ratio = 1;
    res.le = { type: 'expansion', from: Minf, to: M };
  }
  if (!(M > 1)) {
    res.ok = false;
    res.reason = 'flow behind the leading-edge shock is subsonic';
    return res;
  }
  const pinfOverP0 = pOverP0(Minf);
  let nu = prandtlMeyer(M);
  for (let k = 0; k < n; k++) {
    if (k > 0) {
      // smooth surface: isentropic simple-wave turning (expansion or compression)
      nu -= turn[k] - turn[k - 1];
      if (nu <= 0) {
        res.ok = false;
        res.reason = 'isentropic compression drives the surface flow subsonic';
        return res;
      }
      M = inversePrandtlMeyer(nu);
      if (!Number.isFinite(M)) {
        res.ok = false;
        res.reason = 'expansion exceeds the vacuum limit';
        return res;
      }
    }
    mach[k] = M;
    const pr = (pOverP0(M) * p0ratio) / pinfOverP0; // p / p_inf
    cp[k] = cpFromPressureRatio(pr, Minf);
  }
  return res;
}

/** Integrate surface Cp into lift and drag (wind axes) and pitching moment about x/c = 0.25. */
export function integrateForces(pu, cpu, pl, cpl, alpha, xmU, xmL) {
  let fx = 0, fy = 0, m = 0;
  // upper: outward normal (-ty, tx); lower: outward normal (ty, -tx); force = -Cp n ds
  for (let k = 0; k < pu.n; k++) {
    const nx = -pu.ty[k], ny = pu.tx[k];
    const dfx = -cpu[k] * nx * pu.ds[k], dfy = -cpu[k] * ny * pu.ds[k];
    fx += dfx; fy += dfy;
    m -= (xmU[k] - 0.25) * dfy;
  }
  for (let k = 0; k < pl.n; k++) {
    const nx = pl.ty[k], ny = -pl.tx[k];
    const dfx = -cpl[k] * nx * pl.ds[k], dfy = -cpl[k] * ny * pl.ds[k];
    fx += dfx; fy += dfy;
    m -= (xmL[k] - 0.25) * dfy;
  }
  const ca = Math.cos(alpha), sa = Math.sin(alpha);
  return { cl: fy * ca - fx * sa, cd: fx * ca + fy * sa, cm: m };
}

/**
 * Evaluate a shape with shock-expansion theory (and Ackeret for comparison).
 * @param {number[]} shape design vector
 * @param {number} Minf free-stream Mach (> 1)
 * @param {number} alphaDeg angle of attack, degrees
 */
export function evaluateShockExpansion(shape, Minf, alphaDeg, { withLinear = true } = {}) {
  const alpha = (alphaDeg * Math.PI) / 180;
  const s = surfaces(shape, XS);
  const pu = panels(s.x, s.yu);
  const pl = panels(s.x, s.yl);
  const turnU = Float64Array.from(pu.ang, (a) => a - alpha);
  const turnL = Float64Array.from(pl.ang, (a) => alpha - a);
  const tmax = maxDeflection(Minf);
  const up = marchSurface(Minf, turnU, tmax);
  const lo = marchSurface(Minf, turnL, tmax);
  const out = {
    engine: 'shock-expansion',
    valid: up.ok && lo.ok,
    warnings: [],
    x: pu.xm,
    geometry: s,
  };
  if (!up.ok) out.warnings.push(`Upper surface: ${up.reason}.`);
  if (!lo.ok) out.warnings.push(`Lower surface: ${lo.reason}.`);
  if (out.valid) {
    const f = integrateForces(pu, up.cp, pl, lo.cp, alpha, pu.xm, pl.xm);
    Object.assign(out, { cpU: up.cp, cpL: lo.cp, machU: up.mach, machL: lo.mach, ...f });
    // Waves for the flow overlay: Mach angle along each surface and TE turning.
    out.waves = {
      upper: { le: up.le, ang: pu.ang, mach: up.mach, turn: turnU },
      lower: { le: lo.le, ang: pl.ang, mach: lo.mach, turn: turnL },
      alpha,
      Minf,
    };
  }
  if (withLinear) {
    const b = Math.sqrt(Minf * Minf - 1);
    const cpU = Float64Array.from(turnU, (t) => (2 * t) / b);
    const cpL = Float64Array.from(turnL, (t) => (2 * t) / b);
    out.linear = { cpU, cpL, ...integrateForces(pu, cpU, pl, cpL, alpha, pu.xm, pl.xm) };
  }
  return out;
}

export const SHOCK_EXPANSION_STATIONS = N_STATIONS;
export { GAMMA };
