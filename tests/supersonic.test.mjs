import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  prandtlMeyer, inversePrandtlMeyer, obliqueShock, maxDeflection, cpCritical,
} from '../src/gasdynamics.js';
import { evaluateShockExpansion } from '../src/shockExpansion.js';
import { PRESETS, area, maxThickness, surfaces, cosineStations, HANDLE_X } from '../src/geometry.js';

const deg = Math.PI / 180;
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

test('Prandtl-Meyer function matches tables and inverts', () => {
  close(prandtlMeyer(2) / deg, 26.3798, 1e-3, 'nu(2)');
  close(prandtlMeyer(3) / deg, 49.7573, 1e-3, 'nu(3)');
  for (const M of [1.05, 1.5, 2.5, 4, 8]) close(inversePrandtlMeyer(prandtlMeyer(M)), M, 1e-9, `inverse at ${M}`);
});

test('oblique shock matches NACA 1135 values', () => {
  const s = obliqueShock(2, 10 * deg);
  close(s.beta / deg, 39.3139, 1e-3, 'beta');
  close(s.p2p1, 1.7066, 1e-3, 'p2/p1');
  close(s.M2, 1.6405, 1e-3, 'M2');
  close(maxDeflection(2).theta / deg, 22.9735, 1e-3, 'theta_max(M=2)');
  assert.equal(obliqueShock(2, 25 * deg), null, 'detached above theta_max');
});

test('critical Cp at M=0.8', () => {
  close(cpCritical(0.8), -0.4346, 1e-3, 'Cp*');
});

test('geometry: handles interpolate exactly, area is consistent', () => {
  const shape = PRESETS['Biconvex 6%'];
  const xs = Float64Array.from(HANDLE_X);
  const s = surfaces(shape, xs);
  for (let j = 0; j < 5; j++) close(s.yu[j], shape[j], 1e-12, 'handle');
  // biconvex y = 0.12 x(1-x) is exactly in the span -> area = 2 * 0.12 / 6
  close(area(shape), 0.04, 1e-9, 'area');
  close(maxThickness(shape).t, 0.06, 1e-4, 't/c');
  const xf = cosineStations(41);
  const sf = surfaces(shape, xf);
  for (let k = 0; k < xf.length; k++) close(sf.yu[k], 0.12 * xf[k] * (1 - xf[k]), 1e-12, 'exact biconvex');
});

test('flat plate: exact shock-expansion values and Cd = Cl tan(alpha)', () => {
  const flat = new Array(10).fill(0);
  const r = evaluateShockExpansion(flat, 2, 5);
  assert.ok(r.valid);
  // independent hand computation
  const sh = obliqueShock(2, 5 * deg);
  const Mu = inversePrandtlMeyer(prandtlMeyer(2) + 5 * deg);
  const g = 1.4, iso = (M) => Math.pow(1 + 0.2 * M * M, -3.5);
  const pu = iso(Mu) / iso(2);
  const cl = (2 / (g * 4)) * (sh.p2p1 - pu) * Math.cos(5 * deg);
  close(r.cl, cl, 1e-9, 'cl');
  close(r.cd, r.cl * Math.tan(5 * deg), 1e-9, 'cd');
  // Ackeret: cl = 4 alpha / beta
  close(r.linear.cl, ((4 * 5 * deg) / Math.sqrt(3)) * Math.cos(5 * deg), 1e-9, 'Ackeret cl (body-axis normal force rotated to lift)');
});

test('biconvex: Ackeret wave drag and convergence of shock-expansion to it as t -> 0', () => {
  for (const tc of [0.02, 0.06]) {
    const f = (x) => 2 * tc * x * (1 - x);
    const sh = [...[0.1, 0.3, 0.5, 0.7, 0.9].map(f), ...[0.1, 0.3, 0.5, 0.7, 0.9].map((x) => -f(x))];
    const r = evaluateShockExpansion(sh, 2, 0);
    const exact = (16 / 3) * tc * tc / Math.sqrt(3); // Ackeret biconvex
    // panel angles use atan(slope) rather than slope, a <1% O(t^2) difference
    close(r.linear.cd, exact, exact * 1e-2, `Ackeret cd t=${tc}`);
    close(r.cl, 0, 1e-12, 'symmetric => no lift');
    const rel = Math.abs(r.cd - exact) / exact;
    assert.ok(rel < (tc < 0.03 ? 0.02 : 0.06), `shock-expansion within linear-theory error band (rel ${rel})`);
  }
});

test('detached shock is reported, not silently computed', () => {
  const r = evaluateShockExpansion(PRESETS['Biconvex 6%'], 1.15, 4);
  assert.equal(r.valid, false);
  assert.ok(r.warnings.length > 0);
});
