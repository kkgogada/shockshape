import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TSDSolver } from '../src/tsd.js';
import { evaluateShockExpansion } from '../src/shockExpansion.js';

const biconvex = (tc) => {
  const f = (x) => 2 * tc * x * (1 - x);
  const xs = [0.1, 0.3, 0.5, 0.7, 0.9];
  return [...xs.map(f), ...xs.map((x) => -f(x))];
};
const flat = new Array(10).fill(0);
const rel = (a, b) => Math.abs(a - b) / Math.abs(b);

test('subsonic lift matches Prandtl-Glauert thin-airfoil theory', () => {
  const r = new TSDSolver().solve(flat, 0.5, 1);
  const pg = (2 * Math.PI * (Math.PI / 180)) / Math.sqrt(0.75);
  assert.ok(r.solver.converged);
  assert.ok(rel(r.cl, pg) < 0.02, `cl ${r.cl} vs ${pg}`);
});

test('subsonic thickness pressure matches Prandtl-Glauert source solution', () => {
  const tc = 0.02, M = 0.5;
  const r = new TSDSolver().solve(biconvex(tc), M, 0);
  // thin-airfoil theory for y = 2 t x(1-x): u = (2t/pi)[(1-2x) ln(x/(1-x)) + 2] / beta
  const beta = Math.sqrt(1 - M * M);
  let worst = 0;
  for (let k = 0; k < r.x.length; k++) {
    const x = r.x[k];
    if (x < 0.15 || x > 0.85) continue;
    const u = ((2 * tc) / Math.PI) * ((1 - 2 * x) * Math.log(x / (1 - x)) + 2) / beta;
    worst = Math.max(worst, Math.abs(r.cpU[k] + 2 * u));
  }
  assert.ok(worst < 0.004, `max |dCp| ${worst}`);
});

test('shock-free subsonic flow has zero wave drag (d\'Alembert)', () => {
  const r = new TSDSolver().solve(biconvex(0.04), 0.6, 1);
  assert.equal(r.shocks.length, 0);
  assert.equal(r.cd, 0);
  assert.ok(!r.supercritical);
});

test('supersonic limit agrees with Ackeret and shock-expansion theory', () => {
  const shape = biconvex(0.02);
  const r = new TSDSolver().solve(shape, 1.5, 0);
  const ack = (16 / 3) * 0.02 ** 2 / Math.sqrt(1.5 ** 2 - 1);
  const se = evaluateShockExpansion(shape, 1.5, 0).cd;
  assert.ok(rel(r.cd, ack) < 0.03, `cd ${r.cd} vs Ackeret ${ack}`);
  assert.ok(rel(r.cd, se) < 0.05, `cd ${r.cd} vs shock-expansion ${se}`);
});

test('supercritical flow develops an embedded shock and wave drag', () => {
  const r = new TSDSolver().solve(biconvex(0.06), 0.88, 0);
  assert.ok(r.solver.converged);
  assert.ok(r.supercritical, `max local Mach ${r.maxLocalMach}`);
  assert.ok(r.shocks.length > 0);
  assert.ok(r.cd > 1e-3, `cd ${r.cd}`);
});
