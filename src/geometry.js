// Airfoil parameterization: sharp-leading-edge, sharp-trailing-edge CST
// (Kulfan class function C(x) = x(1 - x), i.e. N1 = N2 = 1) with a
// degree-4 Bernstein shape function on each surface.
//
// The design variables are not the raw Bernstein weights. They are the
// surface heights at five fixed "handle" stations, which is what the user
// drags in the editor. Heights map linearly to weights (5x5 solve), so every
// surface ordinate is a linear function of the design vector:
//     y(x) = sum_j L_j(x) * h_j
// which makes geometry, thickness and area constraints cheap and exact.

export const HANDLE_X = [0.1, 0.3, 0.5, 0.7, 0.9];
export const N_HANDLES = HANDLE_X.length;      // per surface
export const N_SHAPE = 2 * N_HANDLES;          // upper + lower
const ORDER = N_HANDLES - 1;                   // Bernstein degree

function binom(n, k) {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

function bernstein(i, n, x) {
  return binom(n, i) * Math.pow(x, i) * Math.pow(1 - x, n - i);
}

function bernsteinDeriv(i, n, x) {
  // d/dx B_{i,n}(x) = n (B_{i-1,n-1} - B_{i,n-1})
  const a = i > 0 ? bernstein(i - 1, n - 1, x) : 0;
  const b = i < n ? bernstein(i, n - 1, x) : 0;
  return n * (a - b);
}

const classFn = (x) => x * (1 - x);
const classFnDeriv = (x) => 1 - 2 * x;

function invert(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const piv = A[c][c];
    for (let k = 0; k < 2 * n; k++) A[c][k] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      for (let k = 0; k < 2 * n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((row) => row.slice(n));
}

// Interpolation matrix: heights at handle stations = M * weights
const M_INTERP = HANDLE_X.map((xh) =>
  Array.from({ length: N_HANDLES }, (_, i) => classFn(xh) * bernstein(i, ORDER, xh))
);
const M_INV = invert(M_INTERP); // weights = M_INV * heights

/** Cardinal basis L_j(x) and its derivative, so y(x) = sum_j L_j(x) h_j. */
export function basis(x) {
  const L = new Float64Array(N_HANDLES);
  const dL = new Float64Array(N_HANDLES);
  const C = classFn(x);
  const dC = classFnDeriv(x);
  for (let i = 0; i < N_HANDLES; i++) {
    const B = bernstein(i, ORDER, x);
    const dB = bernsteinDeriv(i, ORDER, x);
    for (let j = 0; j < N_HANDLES; j++) {
      L[j] += C * B * M_INV[i][j];
      dL[j] += (dC * B + C * dB) * M_INV[i][j];
    }
  }
  return { L, dL };
}

/** Cosine-clustered chordwise stations (dense at LE and TE). */
export function cosineStations(n) {
  return Float64Array.from({ length: n }, (_, k) => 0.5 * (1 - Math.cos((Math.PI * k) / (n - 1))));
}

// Basis values cached per station set: geometry evaluation is then a tiny
// mat-vec, which matters because the optimizer evaluates it thousands of times.
const basisCache = new Map();
function cachedBasis(xs) {
  let c = basisCache.get(xs);
  if (!c) {
    c = Array.from(xs, (x) => basis(x));
    basisCache.set(xs, c);
  }
  return c;
}

/**
 * Evaluate both surfaces.
 * @param {number[]} shape  length-10 design vector [upper h0..h4, lower h0..h4]
 * @param {Float64Array} xs chordwise stations in [0, 1]
 */
export function surfaces(shape, xs) {
  const B = cachedBasis(xs);
  const n = xs.length;
  const yu = new Float64Array(n), yl = new Float64Array(n);
  const su = new Float64Array(n), sl = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const { L, dL } = B[k];
    for (let j = 0; j < N_HANDLES; j++) {
      yu[k] += L[j] * shape[j];
      su[k] += dL[j] * shape[j];
      yl[k] += L[j] * shape[N_HANDLES + j];
      sl[k] += dL[j] * shape[N_HANDLES + j];
    }
  }
  return { x: xs, yu, yl, su, sl };
}

/** Cross-sectional area (per unit chord^2), exact: linear in the design vector. */
const AREA_WEIGHTS = (() => {
  // integrate L_j over [0,1] with composite Simpson on a fine grid
  const n = 2001;
  const w = new Float64Array(N_HANDLES);
  for (let k = 0; k < n; k++) {
    const x = k / (n - 1);
    const c = k === 0 || k === n - 1 ? 1 : k % 2 ? 4 : 2;
    const { L } = basis(x);
    for (let j = 0; j < N_HANDLES; j++) w[j] += (c * L[j]) / (3 * (n - 1));
  }
  return w;
})();

export function area(shape) {
  let a = 0;
  for (let j = 0; j < N_HANDLES; j++) a += AREA_WEIGHTS[j] * (shape[j] - shape[N_HANDLES + j]);
  return a;
}

export function areaGradient() {
  const g = new Float64Array(N_SHAPE);
  for (let j = 0; j < N_HANDLES; j++) {
    g[j] = AREA_WEIGHTS[j];
    g[N_HANDLES + j] = -AREA_WEIGHTS[j];
  }
  return g;
}

const THICK_X = cosineStations(81);
export function maxThickness(shape) {
  const s = surfaces(shape, THICK_X);
  let t = -Infinity, at = 0;
  for (let k = 0; k < THICK_X.length; k++) {
    const d = s.yu[k] - s.yl[k];
    if (d > t) { t = d; at = THICK_X[k]; }
  }
  return { t, x: at };
}

/** Smallest local thickness over the interior (negative => surfaces cross). */
export function minInteriorThickness(shape) {
  const s = surfaces(shape, THICK_X);
  let t = Infinity;
  for (let k = 1; k < THICK_X.length - 1; k++) t = Math.min(t, (s.yu[k] - s.yl[k]) / classFn(THICK_X[k]));
  return t; // normalised by x(1-x) so the sharp ends don't dominate
}

/** Fit handle heights to an arbitrary pair of surface functions (least squares at handles = exact interpolation). */
export function fromFunctions(fu, fl) {
  return [...HANDLE_X.map(fu), ...HANDLE_X.map(fl)];
}

// Preset sections. All are sharp-edged, as supersonic sections must be for an
// attached leading-edge shock.
export const PRESETS = {
  'Biconvex 6%': fromFunctions((x) => 0.12 * x * (1 - x), (x) => -0.12 * x * (1 - x)),
  'Biconvex 4%': fromFunctions((x) => 0.08 * x * (1 - x), (x) => -0.08 * x * (1 - x)),
  'Cambered biconvex 5%': fromFunctions(
    (x) => 0.1 * x * (1 - x) + 0.08 * x * (1 - x),
    (x) => -0.1 * x * (1 - x) + 0.08 * x * (1 - x)
  ),
  'Rounded double-wedge 6%': fromFunctions(
    (x) => 0.03 * Math.sin(Math.PI * x) ** 0.8,
    (x) => -0.03 * Math.sin(Math.PI * x) ** 0.8
  ),
  'Aft-loaded 5%': fromFunctions(
    (x) => 0.025 * 4 * x * (1 - x) * (0.7 + 0.6 * x),
    (x) => -0.025 * 4 * x * (1 - x) * (1.3 - 0.6 * x)
  ),
};
