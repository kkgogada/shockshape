// Compressible-flow relations for a calorically perfect gas.
export const GAMMA = 1.4;
const g = GAMMA;

/** Prandtl-Meyer function nu(M), radians. */
export function prandtlMeyer(M) {
  if (M <= 1) return 0;
  const a = Math.sqrt((g + 1) / (g - 1));
  const b = Math.sqrt(M * M - 1);
  return a * Math.atan(b / a) - Math.atan(b);
}

export const NU_MAX = (Math.PI / 2) * (Math.sqrt((g + 1) / (g - 1)) - 1);

/** Inverse Prandtl-Meyer: Mach number for a given nu (Newton + bisection guard). */
export function inversePrandtlMeyer(nu) {
  if (nu <= 0) return 1;
  if (nu >= NU_MAX) return Infinity;
  let lo = 1, hi = 200, M = 1 + Math.pow(nu, 2 / 3) * 1.3 + nu; // decent start
  if (!(M > lo && M < hi)) M = 2;
  for (let it = 0; it < 60; it++) {
    const f = prandtlMeyer(M) - nu;
    if (Math.abs(f) < 1e-13) break;
    if (f > 0) hi = M; else lo = M;
    const dnu = Math.sqrt(M * M - 1) / (M * (1 + ((g - 1) / 2) * M * M)); // dnu/dM
    let Mn = M - f / dnu;
    if (!(Mn > lo && Mn < hi)) Mn = 0.5 * (lo + hi);
    M = Mn;
  }
  return M;
}

/** p / p0 (isentropic). */
export function pOverP0(M) {
  return Math.pow(1 + ((g - 1) / 2) * M * M, -g / (g - 1));
}

/** Deflection angle theta for shock angle beta at Mach M (theta-beta-M relation). */
export function thetaFromBeta(M, beta) {
  const s = Math.sin(beta);
  const num = 2 * (M * M * s * s - 1) / Math.tan(beta);
  const den = M * M * (g + Math.cos(2 * beta)) + 2;
  return Math.atan(num / den);
}

/** Maximum attached-shock deflection and the shock angle where it occurs. */
export function maxDeflection(M) {
  // golden-section search for max theta on beta in (mu, pi/2)
  let a = Math.asin(1 / M), b = Math.PI / 2;
  const r = (Math.sqrt(5) - 1) / 2;
  let c = b - r * (b - a), d = a + r * (b - a);
  for (let i = 0; i < 80; i++) {
    if (thetaFromBeta(M, c) > thetaFromBeta(M, d)) b = d; else a = c;
    c = b - r * (b - a);
    d = a + r * (b - a);
  }
  const beta = 0.5 * (a + b);
  return { theta: thetaFromBeta(M, beta), beta };
}

/**
 * Weak oblique-shock solution.
 * Returns null if the shock would be detached (theta > theta_max).
 */
export function obliqueShock(M, theta) {
  const mu = Math.asin(1 / M);
  if (theta <= 0) return { beta: mu, M2: M, p2p1: 1, p02p01: 1 };
  const { theta: tmax, beta: bmax } = maxDeflection(M);
  if (theta > tmax) return null;
  // bisection on the weak branch beta in (mu, bmax)
  let lo = mu, hi = bmax;
  for (let i = 0; i < 100; i++) {
    const mid = 0.5 * (lo + hi);
    if (thetaFromBeta(M, mid) < theta) lo = mid; else hi = mid;
  }
  const beta = 0.5 * (lo + hi);
  const Mn1 = M * Math.sin(beta);
  const Mn1s = Mn1 * Mn1;
  const p2p1 = 1 + ((2 * g) / (g + 1)) * (Mn1s - 1);
  const Mn2s = (1 + ((g - 1) / 2) * Mn1s) / (g * Mn1s - (g - 1) / 2);
  const M2 = Math.sqrt(Mn2s) / Math.sin(beta - theta);
  const rho = ((g + 1) * Mn1s) / ((g - 1) * Mn1s + 2);
  const p02p01 = Math.pow(rho, g / (g - 1)) * Math.pow(p2p1, -1 / (g - 1));
  return { beta, M2, p2p1, p02p01 };
}

/** Pressure coefficient from static-pressure ratio p/p_inf. */
export function cpFromPressureRatio(pr, Minf) {
  return (2 / (g * Minf * Minf)) * (pr - 1);
}

/** Critical pressure coefficient (local Mach = 1). */
export function cpCritical(Minf) {
  const r = (2 + (g - 1) * Minf * Minf) / (g + 1);
  return (2 / (g * Minf * Minf)) * (Math.pow(r, g / (g - 1)) - 1);
}
