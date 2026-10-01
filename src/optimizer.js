// Constrained drag minimisation:
//
//     minimise  Cd(shape, alpha)
//     subject to  Cl = Cl_target            (alpha is a design variable)
//                 t/c >= t_min   or   area >= A_min
//                 upper and lower surfaces do not cross
//
// Without a thickness/area constraint the optimum is a zero-thickness flat
// plate, so a constraint is not optional: it is what makes the problem real
// airfoil design.
//
// Method: augmented Lagrangian (method of multipliers) with BFGS inner
// iterations and an Armijo backtracking line search. Gradients come from the
// active engine: exact reverse-mode for the neural surrogate, finite
// differences for the analytic / TSD engines.

import { N_SHAPE, area, areaGradient, surfaces, cosineStations } from './geometry.js';

const TX = cosineStations(61);
const DESIGN_SCALE = [...new Array(N_SHAPE).fill(0.01), 1]; // handle heights ~0.01, alpha ~1 deg

/**
 * Maximum thickness t/c, refined by a parabola through the three stations
 * around the discrete maximum so it varies smoothly as the peak moves.
 */
export function smoothMaxThickness(shape) {
  const s = surfaces(shape, TX);
  let kb = 1, best = -Infinity;
  for (let k = 1; k < TX.length - 1; k++) {
    const d = s.yu[k] - s.yl[k];
    if (d > best) { best = d; kb = k; }
  }
  const x0 = TX[kb - 1], x1 = TX[kb], x2 = TX[kb + 1];
  const f0 = s.yu[kb - 1] - s.yl[kb - 1], f1 = best, f2 = s.yu[kb + 1] - s.yl[kb + 1];
  const d01 = (f1 - f0) / (x1 - x0), d12 = (f2 - f1) / (x2 - x1);
  const a = (d12 - d01) / (x2 - x0);
  if (a >= 0) return best;
  const xs = 0.5 * (x0 + x1) - d01 / (2 * a); // vertex of the interpolating parabola
  const xv = Math.min(x2, Math.max(x0, xs));
  return f0 + d01 * (xv - x0) + a * (xv - x0) * (xv - x1); // Newton-form interpolant
}

function minCrossing(shape) {
  const s = surfaces(shape, TX);
  let m = Infinity;
  for (let k = 1; k < TX.length - 1; k++) m = Math.min(m, (s.yu[k] - s.yl[k]) / (TX[k] * (1 - TX[k])));
  return m;
}

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

export class Optimizer {
  /**
   * @param {object} o
   * @param {(shape:number[], alpha:number, opts?:object)=>object} o.evaluate engine call
   * @param {number[]} o.shape initial shape
   * @param {number} o.alpha initial alpha (deg)
   * @param {number|null} o.clTarget null => alpha held fixed
   * @param {'thickness'|'area'} o.constraint
   * @param {number} o.limit t_min (t/c) or A_min
   * @param {'exact'|'central'|'forward'} o.gradientMode
   */
  constructor(o) {
    this.o = { fdStep: 2e-4, maxIter: 80, ...o };
    this.free = [...new Array(N_SHAPE).fill(true), o.clTarget !== null && o.clTarget !== undefined];
    this.z = [...o.shape, o.alpha];
    this.lam = { cl: 0, geo: 0, cross: 0 };
    this.rho = 10;
    this.iter = 0;
    this.evals = 0;
    this.H = null;
    this.history = [];
    this.lastViolation = Infinity;
    this.done = false;
  }

  // ---------- problem functions (scaled) ----------
  engine(z, gradients = false) {
    this.evals++;
    return this.o.evaluate(z.slice(0, N_SHAPE), z[N_SHAPE], { gradients });
  }

  constraints(z, r) {
    const shape = z.slice(0, N_SHAPE);
    const c = {};
    if (this.free[N_SHAPE]) c.cl = (r.cl - this.o.clTarget) * 10;
    if (this.o.constraint === 'area') c.geo = (this.o.limit - area(shape)) / this.o.limit;
    else c.geo = (this.o.limit - smoothMaxThickness(shape)) / this.o.limit;
    c.cross = -minCrossing(shape) / 0.05; // <= 0 when surfaces are separated
    return c;
  }

  lagrangian(z, r) {
    if (!r.valid || !Number.isFinite(r.cd)) return Infinity;
    const c = this.constraints(z, r);
    let L = r.cd * 100;
    if (c.cl !== undefined) L += this.lam.cl * c.cl + 0.5 * this.rho * c.cl * c.cl;
    for (const k of ['geo', 'cross']) {
      const t = Math.max(0, this.lam[k] + this.rho * c[k]);
      L += (t * t - this.lam[k] * this.lam[k]) / (2 * this.rho);
    }
    return L;
  }

  /** Gradient of the Lagrangian w.r.t. the scaled free variables. */
  gradient(z, r0) {
    const n = z.length;
    const g = new Float64Array(n);
    const L0 = this.lagrangian(z, r0);
    if (this.o.gradientMode === 'exact' && r0.grad) {
      // engine part analytically, geometry part by cheap finite differences
      const c = this.constraints(z, r0);
      const dLdcd = 100;
      const dLdcl = c.cl !== undefined ? (this.lam.cl + this.rho * c.cl) * 10 : 0;
      const geoOnly = (zz) => this.lagrangian(zz, { ...r0, valid: true, cd: r0.cd, cl: r0.cl });
      for (let k = 0; k < n; k++) {
        if (!this.free[k]) continue;
        let gk = dLdcd * r0.grad.cd[k] + dLdcl * r0.grad.cl[k];
        if (k < N_SHAPE) {
          const h = 1e-6, zp = z.slice(), zm = z.slice();
          zp[k] += h; zm[k] -= h;
          gk += (geoOnly(zp) - geoOnly(zm)) / (2 * h); // area/thickness/crossing terms
        }
        g[k] = gk * DESIGN_SCALE[k];
      }
      return { g, L0 };
    }
    for (let k = 0; k < n; k++) {
      if (!this.free[k]) continue;
      const h = this.o.fdStep * (k < N_SHAPE ? 1 : 50); // alpha step in degrees
      const zp = z.slice();
      zp[k] += h;
      const Lp = this.lagrangian(zp, this.engine(zp));
      if (this.o.gradientMode === 'central') {
        const zm = z.slice();
        zm[k] -= h;
        const Lm = this.lagrangian(zm, this.engine(zm));
        g[k] = ((Lp - Lm) / (2 * h)) * DESIGN_SCALE[k];
      } else {
        g[k] = ((Lp - L0) / h) * DESIGN_SCALE[k];
      }
    }
    return { g, L0 };
  }

  clamp(z) {
    for (let k = 0; k < N_SHAPE; k++) z[k] = Math.max(-0.2, Math.min(0.2, z[k]));
    z[N_SHAPE] = Math.max(-8, Math.min(12, z[N_SHAPE]));
    return z;
  }

  record(r, z, extra = {}) {
    const c = this.constraints(z, r);
    const viol = Math.max(Math.abs(c.cl ?? 0) / 10, Math.max(0, c.geo) * this.o.limit, Math.max(0, c.cross) * 0.05);
    const rec = {
      iter: this.iter,
      shape: z.slice(0, N_SHAPE),
      alpha: z[N_SHAPE],
      cd: r.cd,
      cl: r.cl,
      area: area(z.slice(0, N_SHAPE)),
      tc: smoothMaxThickness(z.slice(0, N_SHAPE)),
      violation: viol,
      evals: this.evals,
      lambda: { ...this.lam },
      ...extra,
    };
    this.history.push(rec);
    return rec;
  }

  /** Start: evaluate the initial design. */
  init() {
    this.z = this.clamp(this.z);
    this.r = this.engine(this.z, this.o.gradientMode === 'exact');
    if (!this.r.valid) throw new Error(this.r.warnings?.join(' ') || 'initial design is invalid for this engine');
    this.o.onAccept?.();
    return this.record(this.r, this.z, { phase: 'start' });
  }

  /** One BFGS step on the current augmented Lagrangian (plus multiplier update every few steps). */
  step() {
    if (this.done) return null;
    this.iter++;
    const n = this.z.length;
    const free = this.free;
    const { g, L0 } = this.cached ?? this.gradient(this.z, this.r);
    this.cached = null;
    if (!this.H) {
      this.H = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j && free[i] ? 1 : 0)));
    }
    // direction d = -H g (scaled space)
    const d = this.H.map((row) => -dot(row, g));
    // keep the first step modest: at most 0.6 scaled units (~0.006 chord / 0.6 deg)
    const dn = Math.sqrt(dot(d, d)) || 1;
    let t = Math.min(1, 0.6 / dn);
    const slope = dot(g, d);
    let zNew = null, rNew = null, LNew = Infinity;
    for (let ls = 0; ls < 10; ls++) {
      const zt = this.clamp(this.z.map((v, k) => v + t * d[k] * DESIGN_SCALE[k]));
      const rt = this.engine(zt, this.o.gradientMode === 'exact');
      const Lt = this.lagrangian(zt, rt);
      if (Lt <= L0 + 1e-4 * t * slope) { zNew = zt; rNew = rt; LNew = Lt; break; }
      t *= 0.5;
    }
    if (!zNew) {
      // no decrease along the quasi-Newton direction: reset curvature, update multipliers
      this.H = null;
      this.updateMultipliers(this.z, this.r);
      return this.record(this.r, this.z, { phase: 'reset', L: L0 });
    }
    this.o.onAccept?.();
    // BFGS update in scaled coordinates
    const s = zNew.map((v, k) => (v - this.z[k]) / DESIGN_SCALE[k]);
    const gradNew = this.gradient(zNew, rNew);
    const gNew = gradNew.g;
    const yv = Array.from(gNew, (v, k) => v - g[k]);
    const sy = dot(s, yv);
    if (sy > 1e-10) {
      const Hy = this.H.map((row) => dot(row, yv));
      const yHy = dot(yv, Hy);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          this.H[i][j] += ((sy + yHy) * s[i] * s[j]) / (sy * sy) - (Hy[i] * s[j] + s[i] * Hy[j]) / sy;
    }
    this.z = zNew;
    this.r = rNew;
    this.cached = gradNew;
    if (this.iter % 4 === 0) this.updateMultipliers(zNew, rNew);
    const rec = this.record(rNew, zNew, { phase: 'step', L: LNew, step: t * dn });
    if (this.iter >= this.o.maxIter || (t * dn < 2e-3 && rec.violation < 2e-4 && this.iter > 8)) this.done = true;
    return rec;
  }

  updateMultipliers(z, r) {
    const c = this.constraints(z, r);
    if (c.cl !== undefined) this.lam.cl += this.rho * c.cl;
    for (const k of ['geo', 'cross']) this.lam[k] = Math.max(0, this.lam[k] + this.rho * c[k]);
    const viol = Math.max(Math.abs(c.cl ?? 0), Math.max(0, c.geo), Math.max(0, c.cross));
    if (viol > 0.5 * this.lastViolation) this.rho = Math.min(this.rho * 2, 1e4);
    this.lastViolation = viol;
    this.H = null; // the merit function changed
    this.cached = null;
  }
}
