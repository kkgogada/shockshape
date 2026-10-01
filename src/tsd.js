// Transonic small-disturbance (TSD) solver.
//
//   d/dx [ K u - (g+1)/2 M^2 u^2 ] + d/dy [ phi_y ] = 0,   u = phi_x,  K = 1 - M^2
//
// Discretised with Murman's fully conservative type-dependent scheme
// (central differences at subsonic points, upwind at supersonic points,
// with shock-point and sonic-point operators), and solved by Newton-
// linearised successive line over-relaxation (SLOR) on vertical lines,
// sweeping downstream. Thin-airfoil boundary conditions are transferred
// to y = 0 +/- , lift enters through a wake cut carrying the circulation
// jump (Kutta condition), and the subsonic far field is a compressible
// point vortex.
//
// Reference: Murman & Cole, AIAA J. 9(1), 1971; Murman, AIAA J. 12(5), 1974.

import { surfaces } from './geometry.js';
import { GAMMA } from './gasdynamics.js';

function stretchedOutward(start, h0, ratio, limit, dir) {
  // nodes beyond `start`, first spacing h0, growing geometrically to |limit|
  const out = [];
  let x = start, h = h0;
  while (dir > 0 ? x < limit : x > limit) {
    h *= ratio;
    x += dir * h;
    out.push(x);
  }
  return out;
}

export function buildGrid({ na = 48, ratio = 1.13, xFar = 40, yFar = 40, dyFactor = 0.6 } = {}) {
  const h = 1 / na;
  const airfoil = Array.from({ length: na }, (_, k) => (k + 0.5) * h);
  const ahead = stretchedOutward(airfoil[0], h, ratio, -xFar, -1).reverse();
  const behind = stretchedOutward(airfoil[na - 1], h, ratio, 1 + xFar, 1);
  const x = Float64Array.from([...ahead, ...airfoil, ...behind]);
  const dy0 = h * dyFactor;
  const yPos = [0.5 * dy0];
  let dy = dy0;
  while (yPos[yPos.length - 1] < yFar) {
    dy *= ratio;
    yPos.push(yPos[yPos.length - 1] + dy);
  }
  const y = Float64Array.from([...yPos.map((v) => -v).reverse(), ...yPos]);
  const NI = x.length, NJ = y.length;
  const jU = yPos.length, jL = jU - 1; // rows just above / below the cut
  let iLE = -1, iTE = -1;
  for (let i = 0; i < NI; i++) {
    if (x[i] > 0 && x[i] < 1) {
      if (iLE < 0) iLE = i;
      iTE = i;
    }
  }
  return { x, y, NI, NJ, jU, jL, iLE, iTE, airfoilX: Float64Array.from(airfoil) };
}

function solveSmall(M, b) {
  const n = b.length;
  const A = M.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-300) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = A[r][n];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x;
}

/** One grid level of the solver. */
class TSDLevel {
  constructor(gridOpts) {
    this.g = buildGrid(gridOpts);
    const { NI, NJ } = this.g;
    this.phi = new Float64Array(NI * NJ);
    this.gamma = 0;
    this.lastKey = null;
    // scratch for the tridiagonal solves
    this.a = new Float64Array(NJ);
    this.b = new Float64Array(NJ);
    this.c = new Float64Array(NJ);
    this.r = new Float64Array(NJ);
    this.mu = new Uint8Array(NI * NJ);
    this.colOld = new Float64Array(NJ);
  }

  reset() {
    this.phi.fill(0);
    this.gamma = 0;
  }

  /** Relax on this grid from the current iterate. */
  relax(shape, Minf, alphaDeg, { tol = 1e-7, maxIter = 6000, omega = 1.5, superFloor = 0.05, duMax = 0.05, symmetric = true, omegaGamma = 1, aaDepth = 6, aaEvery = 4, deadline = 0, cold = false, onProgress } = {}) {
    const g = this.g;
    const { x, y, NI, NJ, jU, jL, iLE, iTE } = g;
    const phi = this.phi;
    if (cold) this.reset();
    const alpha = (alphaDeg * Math.PI) / 180;
    const K = 1 - Minf * Minf;
    const cq = ((GAMMA + 1) / 2) * Minf * Minf; // F(u) = K u - cq u^2
    const F = (u) => K * u - cq * u * u;
    const dF = (u) => K - 2 * cq * u;
    const supersonicFree = K < 0;
    if (supersonicFree) {
      // Supersonic free stream: marching dominates; over-relaxing the
      // subsonic pocket behind a detached bow shock, or Anderson mixing
      // across it, makes the near-sonic cases cycle.
      omega = Math.min(omega, 1.0);
      aaDepth = 0;
    }
    const beta = Math.sqrt(Math.max(K, 0.01));

    // boundary conditions phi_y(0+/-) = slope - alpha on the airfoil
    const s = surfaces(shape, g.airfoilX);
    const bcU = new Float64Array(NI), bcL = new Float64Array(NI);
    for (let i = iLE; i <= iTE; i++) {
      bcU[i] = s.su[i - iLE] - alpha;
      bcL[i] = s.sl[i - iLE] - alpha;
    }

    const id = (i, j) => i * NJ + j;
    const farField = (gam) => {
      if (supersonicFree) {
        // undisturbed upstream / top / bottom; outflow handled by extrapolation
        for (let j = 0; j < NJ; j++) { phi[id(0, j)] = 0; phi[id(1, j)] = 0; }
        for (let i = 0; i < NI; i++) { phi[id(i, 0)] = 0; phi[id(i, NJ - 1)] = 0; }
        return;
      }
      const v = (xx, yy) => -(gam / (2 * Math.PI)) * Math.atan2(-beta * yy, -(xx - 0.25));
      for (let j = 0; j < NJ; j++) {
        phi[id(0, j)] = v(x[0], y[j]);
        phi[id(1, j)] = v(x[1], y[j]);
        phi[id(NI - 1, j)] = v(x[NI - 1], y[j]);
      }
      for (let i = 0; i < NI; i++) {
        phi[id(i, 0)] = v(x[i], y[0]);
        phi[id(i, NJ - 1)] = v(x[i], y[NJ - 1]);
      }
    };

    const muAt = (i, j) => {
      if (i <= 0 || i >= NI - 1) return supersonicFree ? 1 : 0;
      const uc = (phi[id(i + 1, j)] - phi[id(i - 1, j)]) / (x[i + 1] - x[i - 1]);
      return dF(uc) < 0 ? 1 : 0;
    };

    const { a, b, c, r } = this;
    let iter = 0, corr = Infinity, history = [], limitCycle = false, timedOut = false;
    const gammaHist = [];
    this.aa = null;
    this.aaDisabled = false;
    farField(this.gamma);
    for (iter = 1; iter <= maxIter; iter++) {
      let maxCorr = 0;
      // Point types are frozen for the sweep, from the current iterate.
      const MU = this.mu;
      for (let i = 0; i < NI; i++) for (let j = 0; j < NJ; j++) MU[id(i, j)] = muAt(i, j);
      // Subsonic free stream: alternate downstream and upstream sweeps so
      // far-field / circulation information travels both ways quickly.
      const backward = symmetric && !supersonicFree && iter % 2 === 0;
      for (let ii = 2; ii <= NI - 2; ii++) {
        const i = backward ? NI - ii : ii;
        const hp = x[i + 1] - x[i], hm = x[i] - x[i - 1], hmm = x[i - 1] - x[i - 2];
        const dxi = 0.5 * (x[i + 1] - x[i - 1]);
        const dxim = 0.5 * (x[i] - x[i - 2]);
        const onAirfoil = i >= iLE && i <= iTE;
        const inWake = i > iTE;
        // Columns touching supersonic points get a few inner Newton passes:
        // the residual is quadratic in phi_i, and on early sweeps the old
        // phi_i can be far from the marched value.
        let colSuper = false;
        for (let j = 1; j <= NJ - 2 && !colSuper; j++) colSuper = MU[id(i, j)] === 1 || MU[id(i - 1, j)] === 1;
        const early = iter <= 10;
        const nInner = colSuper ? (early ? 3 : 2) : 1;
        const old = this.colOld;
        for (let j = 0; j < NJ; j++) old[j] = phi[id(i, j)];
        if (colSuper && early) {
          // predictor at fully supersonic points: carry the upstream gradient
          for (let j = 1; j <= NJ - 2; j++) {
            if (MU[id(i, j)] === 1 && MU[id(i - 1, j)] === 1) {
              phi[id(i, j)] = phi[id(i - 1, j)] + (hm * (phi[id(i - 1, j)] - phi[id(i - 2, j)])) / hmm;
            }
          }
        }
        for (let inner = 0; inner < nInner; inner++) {
        for (let j = 1; j <= NJ - 2; j++) {
          const p0 = phi[id(i, j)];
          const up = (phi[id(i + 1, j)] - p0) / hp;
          const um = (p0 - phi[id(i - 1, j)]) / hm;
          const umm = (phi[id(i - 1, j)] - phi[id(i - 2, j)]) / hmm;
          const mui = MU[id(i, j)];
          const muim = MU[id(i - 1, j)];
          // streamwise part: (1 - mu_i) P_i + mu_{i-1} P_{i-1}
          let X = 0, dX = 0;
          if (!mui) {
            X += (F(up) - F(um)) / dxi;
            dX -= (Math.max(dF(up), superFloor) / hp + Math.max(dF(um), superFloor) / hm) / dxi;
          }
          if (muim) {
            X += (F(um) - F(umm)) / dxim;
            // keep the hyperbolic operator well-posed for the line solve
            dX -= Math.max(Math.abs(dF(um)), superFloor) / hm / dxim;
          }
          // normal part with cut / wall treatment
          const yj = y[j];
          let fluxUp, fluxDn, dQup, dQdn, hy;
          let dQdj = 0, wall = false;
          const dyU = y[j + 1] - yj, dyD = yj - y[j - 1];
          hy = 0.5 * (y[j + 1] - y[j - 1]);
          fluxUp = (phi[id(i, j + 1)] - p0) / dyU;
          fluxDn = (p0 - phi[id(i, j - 1)]) / dyD;
          dQup = 1 / dyU;
          dQdn = 1 / dyD;
          if (j === jU && (onAirfoil || inWake)) {
            if (onAirfoil) {
              fluxDn = bcU[i];
              dQdn = 0;
              hy = 0.5 * (y[j] + y[j + 1]);
              dQdj = -1 / dyU;
              wall = true;
            } else {
              fluxDn = (p0 - phi[id(i, j - 1)] - this.gamma) / dyD;
            }
          } else if (j === jL && (onAirfoil || inWake)) {
            if (onAirfoil) {
              fluxUp = bcL[i];
              dQup = 0;
              hy = -0.5 * (y[j] + y[j - 1]);
              dQdj = -1 / dyD;
              wall = true;
            } else {
              fluxUp = (phi[id(i, j + 1)] - this.gamma - p0) / dyU;
            }
          }
          if (!wall) dQdj = -dQup - dQdn;
          const Q = (fluxUp - fluxDn) / hy;
          r[j] = -(X + Q);
          b[j] = dX + dQdj / hy;
          c[j] = dQup / hy;
          a[j] = dQdn / hy;
        }
        // Thomas algorithm on j = 1..NJ-2 (corrections vanish on the boundaries)
        const n0 = 1, n1 = NJ - 2;
        a[n0] = 0;
        c[n1] = 0;
        for (let j = n0 + 1; j <= n1; j++) {
          const m = a[j] / b[j - 1];
          b[j] -= m * c[j - 1];
          r[j] -= m * r[j - 1];
        }
        r[n1] /= b[n1];
        for (let j = n1 - 1; j >= n0; j--) r[j] = (r[j] - c[j] * r[j + 1]) / b[j];
        for (let j = n0; j <= n1; j++) {
          const w = MU[id(i, j)] || inner > 0 ? 1 : omega;
          phi[id(i, j)] += w * r[j];
        }
        }
        // limiter: no sweep may change the local velocity by more than duMax
        // (keeps early sweeps from jumping across the sonic point)
        const cap = duMax * hm;
        for (let j = 1; j <= NJ - 2; j++) {
          let d = phi[id(i, j)] - old[j];
          if (d > cap) d = cap; else if (d < -cap) d = -cap;
          phi[id(i, j)] = old[j] + d;
          if (Math.abs(d) > maxCorr) maxCorr = Math.abs(d);
        }
      }
      if (supersonicFree) for (let j = 0; j < NJ; j++) phi[id(NI - 1, j)] = phi[id(NI - 2, j)];
      // Kutta condition: circulation = potential jump at the trailing edge
      const phiU0 = phi[id(iTE, jU)] - y[jU] * bcU[iTE];
      const phiL0 = phi[id(iTE, jL)] - y[jL] * bcL[iTE];
      this.gamma += omegaGamma * (phiU0 - phiL0 - this.gamma);
      if (aaDepth > 0 && !this.aaDisabled && iter % aaEvery === 0) this.andersonStep(aaDepth);
      farField(this.gamma);
      corr = maxCorr;
      if (iter % 20 === 0) {
        history.push(corr);
        gammaHist.push(this.gamma);
        if (onProgress) onProgress(iter, corr);
        // A shock point can flip between two neighbouring cells forever
        // (a classic limit cycle of type-dependent schemes): the max
        // correction then stalls while the solution itself has settled.
        // Accept it when the circulation has been steady for 400 sweeps.
        const n = gammaHist.length;
        if (iter >= 600 && corr < 5e-3 && n > 20) {
          let lo = Infinity, hi = -Infinity;
          for (let k = n - 20; k < n; k++) { lo = Math.min(lo, gammaHist[k]); hi = Math.max(hi, gammaHist[k]); }
          if (hi - lo < 2e-5 * Math.max(0.05, Math.abs(this.gamma))) { limitCycle = true; break; }
        }
      }
      if (!Number.isFinite(corr) || Math.abs(this.gamma) > 2) { corr = Infinity; break; }
      if (deadline && (iter & 15) === 0 && Date.now() > deadline) { timedOut = true; break; }
      if (corr < tol) break;
    }
    const converged = corr < tol || limitCycle;
    this.last = { bcU, bcL, s, info: { iter, corr, converged, limitCycle: limitCycle && !(corr < tol), timedOut, history } };
    return this.last.info;
  }

  /**
   * Anderson acceleration of the relaxation map (applied to blocks of
   * sweeps). SLOR damps local errors fast but the global modes (circulation,
   * shock position) slowly; Anderson mixing extrapolates along the last few
   * update directions and removes most of that slow tail.
   */
  andersonStep(m) {
    const phi = this.phi;
    const N = phi.length + 1;
    const cur = new Float64Array(N);
    cur.set(phi);
    cur[N - 1] = this.gamma * 10; // weight the circulation like a field value
    if (!this.aa) {
      this.aa = { x: cur, dF: [], dG: [], fPrev: null, gPrev: null, fNormPrev: Infinity, safe: null, rejects: 0 };
      return;
    }
    const A = this.aa;
    const f = new Float64Array(N);
    let fn = 0;
    for (let k = 0; k < N; k++) { f[k] = cur[k] - A.x[k]; fn += f[k] * f[k]; }
    fn = Math.sqrt(fn);
    if (!(fn < 3 * A.fNormPrev) && A.safe) {
      // the last extrapolation made things worse: go back to the plain
      // relaxation iterate it was built from and start the history afresh
      phi.set(A.safe.subarray(0, N - 1));
      this.gamma = A.safe[N - 1] / 10;
      this.aa = { x: A.safe.slice(), dF: [], dG: [], fPrev: null, gPrev: null, fNormPrev: Infinity, safe: null, rejects: (A.rejects || 0) + 1 };
      if (this.aa.rejects > 6) this.aaDisabled = true;
      return;
    }
    if (A.fPrev) {
      const dF = new Float64Array(N), dG = new Float64Array(N);
      for (let k = 0; k < N; k++) { dF[k] = f[k] - A.fPrev[k]; dG[k] = cur[k] - A.gPrev[k]; }
      A.dF.push(dF); A.dG.push(dG);
      if (A.dF.length > m) { A.dF.shift(); A.dG.shift(); }
    }
    A.fPrev = f; A.gPrev = cur.slice(); A.fNormPrev = fn;
    A.safe = cur.slice();
    let next = cur;
    const q = A.dF.length;
    if (q > 0) {
      // least squares: min || f - dF gamma ||  (normal equations, tiny ridge)
      const M = Array.from({ length: q }, () => new Float64Array(q));
      const rhs = new Float64Array(q);
      for (let a = 0; a < q; a++) {
        for (let b = a; b < q; b++) {
          let sum = 0; const u = A.dF[a], v = A.dF[b];
          for (let k = 0; k < N; k++) sum += u[k] * v[k];
          M[a][b] = M[b][a] = sum;
        }
        let r = 0; const u = A.dF[a];
        for (let k = 0; k < N; k++) r += u[k] * f[k];
        rhs[a] = r;
      }
      let tr = 0; for (let a = 0; a < q; a++) tr += M[a][a];
      for (let a = 0; a < q; a++) M[a][a] += 1e-10 * tr + 1e-30;
      const gam = solveSmall(M, rhs);
      if (gam) {
        next = cur.slice();
        for (let a = 0; a < q; a++) {
          const g = gam[a], dG = A.dG[a];
          for (let k = 0; k < N; k++) next[k] -= g * dG[k];
        }
        phi.set(next.subarray(0, N - 1));
        this.gamma = next[N - 1] / 10;
      }
    }
    A.x = next.slice();
  }

  /** Initialise this level by bilinear interpolation from a coarser one (halves kept separate across the cut). */
  interpolateFrom(src) {
    const { x, y, NI, NJ, jU } = this.g;
    const sx = src.g.x, sy = src.g.y, sNJ = src.g.NJ, sjU = src.g.jU;
    const bracket = (arr, lo, hi, v) => {
      if (v <= arr[lo]) return [lo, lo, 0];
      if (v >= arr[hi]) return [hi, hi, 0];
      let a = lo, b = hi;
      while (b - a > 1) { const m = (a + b) >> 1; if (arr[m] <= v) a = m; else b = m; }
      return [a, b, (v - arr[a]) / (arr[b] - arr[a])];
    };
    for (let i = 0; i < NI; i++) {
      const [i0, i1, tx] = bracket(sx, 0, sx.length - 1, x[i]);
      for (let j = 0; j < NJ; j++) {
        const upper = j >= jU;
        const [j0, j1, ty] = upper ? bracket(sy, sjU, sNJ - 1, y[j]) : bracket(sy, 0, sjU - 1, y[j]);
        const p = src.phi;
        const v0 = p[i0 * sNJ + j0] * (1 - ty) + p[i0 * sNJ + j1] * ty;
        const v1 = p[i1 * sNJ + j0] * (1 - ty) + p[i1 * sNJ + j1] * ty;
        this.phi[i * NJ + j] = v0 * (1 - tx) + v1 * tx;
      }
    }
    this.gamma = src.gamma;
  }

  postProcess(shape, Minf, alphaDeg) {
    const { bcU, bcL, s, info } = this.last;
    const { x, y, NI, NJ, jU, jL, iLE, iTE } = this.g;
    const phi = this.phi;
    const id = (i, j) => i * NJ + j;
    const alpha = (alphaDeg * Math.PI) / 180;
    // surface potential extrapolated to y = 0 +/-
    const nA = iTE - iLE + 1;
    const fU = new Float64Array(nA), fL = new Float64Array(nA);
    for (let k = 0; k < nA; k++) {
      const i = iLE + k;
      fU[k] = phi[id(i, jU)] - y[jU] * bcU[i];
      fL[k] = phi[id(i, jL)] - y[jL] * bcL[i];
    }
    const xa = this.g.airfoilX;
    const deriv = (f, k) => {
      if (k === 0) return (f[1] - f[0]) / (xa[1] - xa[0]);
      if (k === nA - 1) return (f[nA - 1] - f[nA - 2]) / (xa[nA - 1] - xa[nA - 2]);
      return (f[k + 1] - f[k - 1]) / (xa[k + 1] - xa[k - 1]);
    };
    const cpU = new Float64Array(nA), cpL = new Float64Array(nA);
    for (let k = 0; k < nA; k++) {
      cpU[k] = -2 * deriv(fU, k);
      cpL[k] = -2 * deriv(fL, k);
    }
    // body-axis forces (thin-airfoil surface integration, midpoint rule on uniform stations)
    const h = 1 / nA;
    let fx = 0, fy = 0, m = 0;
    for (let k = 0; k < nA; k++) {
      fy += (cpL[k] - cpU[k]) * h;
      fx += (cpU[k] * s.su[k] - cpL[k] * s.sl[k]) * h;
      m -= (xa[k] - 0.25) * (cpL[k] - cpU[k]) * h;
    }
    // The pointwise surface integral of lift under-resolves the 1/sqrt(x)
    // leading-edge singularity; the potential jump (Kutta-Joukowski,
    // normal force = 2 * Gamma) is exact for the discrete solution.
    const fySurface = fy;
    fy = 2 * this.gamma;
    const ca = Math.cos(alpha), sa = Math.sin(alpha);
    const cl = fy * ca - fx * sa;

    // Field quantities and shock detection
    const K = 1 - Minf * Minf;
    const cq = (GAMMA + 1) * Minf * Minf;
    const u = new Float64Array(NI * NJ);
    const machField = new Float32Array(NI * NJ);
    for (let i = 1; i < NI - 1; i++) {
      for (let j = 0; j < NJ; j++) {
        const uu = (phi[id(i + 1, j)] - phi[id(i - 1, j)]) / (x[i + 1] - x[i - 1]);
        u[id(i, j)] = uu;
        machField[id(i, j)] = Math.sqrt(Math.max(0, Minf * Minf + cq * uu));
      }
    }
    // Oswatitsch wave drag from shock jumps: Cd = (g+1) M^4 / 6 * int [u]^3 dy
    const shocks = [];
    let shockIntegral = 0;
    for (let j = 0; j < NJ; j++) {
      const hy = j === 0 || j === NJ - 1 ? 0 : 0.5 * (y[j + 1] - y[j - 1]);
      for (let i = 3; i < NI - 3; i++) {
        if (this.mu[id(i - 1, j)] === 1 && this.mu[id(i, j)] === 0) {
          const uUp = Math.max(u[id(i - 1, j)], u[id(i - 2, j)]);
          const uDn = Math.min(u[id(i, j)], u[id(i + 1, j)]);
          const jump = uUp - uDn;
          if (jump > 0) {
            shockIntegral += jump * jump * jump * hy;
            shocks.push({ x: 0.5 * (x[i - 1] + x[i]), y: y[j], jump });
          }
        }
      }
    }
    const cdShock = ((GAMMA + 1) * Minf ** 4 / 6) * shockIntegral;
    const cdSurface = fx * ca + fy * sa;
    // Below M = 1 the thin-airfoil surface integral misses leading-edge suction,
    // so wave drag is taken from the shock entropy jump (zero if shock-free,
    // as d'Alembert requires). Above M = 1 the surface integral is the wave drag.
    const cd = K > 0 ? cdShock : cdSurface;
    let maxLocalMach = 0;
    for (let k = 0; k < NI * NJ; k++) if (machField[k] > maxLocalMach) maxLocalMach = machField[k];
    const warnings = [];
    if (!info.converged) warnings.push(`TSD solver did not fully converge (max correction ${info.corr.toExponential(1)} after ${info.iter} sweeps).`);
    return {
      engine: 'tsd',
      valid: Number.isFinite(cl) && Number.isFinite(cd),
      warnings,
      x: xa,
      cpU, cpL,
      cl, cd, cm: m,
      cdShock, cdSurface, clSurface: fySurface * ca - fx * sa,
      gammaCirc: this.gamma,
      geometry: s,
      solver: info,
      field: { x, y, NI, NJ, mach: machField, mu: this.mu.slice() },
      shocks,
      maxLocalMach,
      supercritical: maxLocalMach > 1 && K > 0,
    };
  }
}

/**
 * TSD solver with grid sequencing: a cold start is converged on coarse grids
 * first (cheap, and it settles the slowly-converging circulation), then
 * interpolated to the fine grid. Warm starts (small design changes in the
 * optimizer) relax on the fine grid only.
 */
export class TSDSolver {
  constructor({ na = 48, levels = 3, ...rest } = {}) {
    this.levels = [];
    for (let l = levels - 1; l >= 0; l--) this.levels.push(new TSDLevel({ na: Math.max(8, na >> l), ...rest }));
    this.warmKey = null;
  }

  get fine() { return this.levels[this.levels.length - 1]; }
  get g() { return this.fine.g; }

  reset() {
    for (const l of this.levels) l.reset();
    this.warmKey = null;
  }

  solve(shape, Minf, alphaDeg, opts = {}) {
    const key = `${Minf.toFixed(3)}`;
    const cold = opts.cold || this.warmKey === null || Math.abs(parseFloat(this.warmKey) - Minf) > 0.02;
    const t0 = Date.now();
    let total = 0;
    opts = { ...opts, deadline: opts.timeBudgetMs ? t0 + opts.timeBudgetMs : 0 };
    // Relax one level; if Anderson mixing blew up, restart that level from
    // where it began with plain relaxation (slower, but robust).
    const relaxSafely = (level) => {
      const start = { phi: level.phi.slice(), gamma: level.gamma };
      let info = level.relax(shape, Minf, alphaDeg, { ...opts, cold: false });
      total += info.iter;
      if (!info.converged && !(info.corr < 1e-3) && !info.timedOut) {
        level.phi.set(start.phi);
        level.gamma = start.gamma;
        info = level.relax(shape, Minf, alphaDeg, { ...opts, cold: false, aaDepth: 0, omega: Math.min(opts.omega ?? 1.5, 1.3) });
        total += info.iter;
        info.fallback = true;
      }
      return info;
    };
    const coldSolve = () => {
      this.levels[0].reset();
      for (let l = 0; l < this.levels.length; l++) {
        if (l > 0) this.levels[l].interpolateFrom(this.levels[l - 1]);
        relaxSafely(this.levels[l]);
      }
    };
    let usedCold = cold;
    if (cold) {
      coldSolve();
    } else {
      const info = relaxSafely(this.fine);
      if (!info.converged) {
        // a warm start from a very different flow (big change in alpha or
        // shape) can land on the wrong side of a shock; start again from scratch
        opts = { ...opts, deadline: opts.timeBudgetMs ? Date.now() + opts.timeBudgetMs : 0 };
        coldSolve();
        usedCold = true;
      }
    }
    this.warmKey = key;
    const res = this.fine.postProcess(shape, Minf, alphaDeg);
    res.solver.totalSweeps = total;
    res.solver.ms = Date.now() - t0;
    res.solver.coldStart = usedCold;
    if (res.solver.timedOut) res.warnings.unshift(`Stopped at the ${(opts.timeBudgetMs / 1000).toFixed(0)} s time budget before converging; the result is approximate.`);
    if (res.supercritical) res.warnings.push('TSD is an isentropic potential model: with shocks it tends to over-predict shock strength and lift relative to Euler, and strongly lifting transonic cases can even have more than one solution.');
    return res;
  }
}
