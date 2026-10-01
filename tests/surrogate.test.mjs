import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Surrogate } from '../src/surrogate.js';
import { evaluateShockExpansion } from '../src/shockExpansion.js';
import { Optimizer } from '../src/optimizer.js';
import { PRESETS } from '../src/geometry.js';

const model = JSON.parse(readFileSync(new URL('../models/surrogate.json', import.meta.url)));
const S = new Surrogate(model);

test('back-propagated gradients match finite differences of the network', () => {
  const shape = PRESETS['Cambered biconvex 5%'];
  const r = S.evaluate(shape, 2.0, 2.5, { gradients: true });
  const h = 1e-6;
  for (let k = 0; k <= 10; k++) {
    const sp = shape.slice(), sm = shape.slice();
    let ap = 2.5, am = 2.5;
    if (k < 10) { sp[k] += h; sm[k] -= h; } else { ap += h; am -= h; }
    const fd = (S.evaluate(sp, 2.0, ap).cd - S.evaluate(sm, 2.0, am).cd) / (2 * h);
    assert.ok(Math.abs(fd - r.grad.cd[k]) < 1e-6 + 1e-4 * Math.abs(fd), `dCd/dz${k}: ${r.grad.cd[k]} vs ${fd}`);
  }
});

test('surrogate reproduces its training physics inside the domain', () => {
  let worst = 0;
  for (const [name, shape] of Object.entries(PRESETS)) {
    for (const M of [1.6, 2.2, 2.8]) {
      for (const a of [0, 2, 4]) {
        const ref = evaluateShockExpansion(shape, M, a, { withLinear: false });
        if (!ref.valid) continue;
        const s = S.evaluate(shape, M, a);
        worst = Math.max(worst, Math.abs(s.cd - ref.cd));
        assert.ok(Math.abs(s.cl - ref.cl) < 0.01, `${name} M${M} a${a}: cl ${s.cl} vs ${ref.cl}`);
      }
    }
  }
  assert.ok(worst < 1.5e-3, `worst |dCd| ${worst}`);
});

test('surrogate-driven optimization is confirmed by the true model', () => {
  const evaluate = (shape, alpha, o) => S.evaluate(shape, 2.0, alpha, o);
  const opt = new Optimizer({
    evaluate, shape: PRESETS['Cambered biconvex 5%'], alpha: 2, clTarget: 0.15,
    constraint: 'thickness', limit: 0.05, gradientMode: 'exact',
  });
  let rec = opt.init();
  while (!opt.done) rec = opt.step();
  const truthEnd = evaluateShockExpansion(rec.shape, 2.0, rec.alpha, { withLinear: false });
  assert.ok(truthEnd.valid);
  assert.ok(Math.abs(truthEnd.cl - 0.15) < 0.01, `true cl ${truthEnd.cl}`);
  // the baseline here is at a lower Cl, so compare against the analytic optimum instead:
  // linear theory for fixed t/c: cd = beta*cl^2/4 + 4 t^2/beta  (double wedge)
  const b = Math.sqrt(3);
  const ideal = (b * 0.15 ** 2) / 4 + (4 * 0.05 ** 2) / b;
  assert.ok(truthEnd.cd < ideal * 1.1, `true optimized cd ${truthEnd.cd} vs ideal ${ideal}`);
  assert.ok(Math.abs(rec.cd - truthEnd.cd) < 1e-3, `surrogate ${rec.cd} vs truth ${truthEnd.cd}`);
});
