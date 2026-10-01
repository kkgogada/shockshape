// Command-line access to the app's physics models.
//   node tools/evaluate.mjs --engine shock-expansion --mach 2 --alpha 2 --shape "0.0108,0.0252,0.03,0.0252,0.0108,-0.0108,-0.0252,-0.03,-0.0252,-0.0108"
//   node tools/evaluate.mjs --preset "Biconvex 6%" --mach 0.85 --alpha 1 --engine tsd --json
import { readFileSync } from 'node:fs';
import { Engines } from '../src/engines.js';
import { PRESETS } from '../src/geometry.js';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => (a.startsWith('--') ? [...acc, [a.slice(2), arr[i + 1]?.startsWith('--') || arr[i + 1] === undefined ? true : arr[i + 1]]] : acc), [])
);
const shape = args.shape ? args.shape.split(',').map(Number) : PRESETS[args.preset ?? 'Biconvex 6%'];
if (!shape || shape.length !== 10) throw new Error('need --shape with 10 comma-separated handle heights, or a valid --preset');
let model = null;
try { model = JSON.parse(readFileSync(new URL('../models/surrogate.json', import.meta.url))); } catch { /* optional */ }
const E = new Engines({ surrogateModel: model });
const r = E.evaluate(args.engine ?? 'auto', shape, +(args.mach ?? 2), +(args.alpha ?? 0), { withLinear: true, timeBudgetMs: 60000 });
const out = {
  engine: r.engine, valid: r.valid, cl: r.cl, cd: r.cd, cm: r.cm, warnings: r.warnings,
  x: Array.from(r.x ?? []), cpU: Array.from(r.cpU ?? []), cpL: Array.from(r.cpL ?? []),
  linear: r.linear ? { cl: r.linear.cl, cd: r.linear.cd, cpU: Array.from(r.linear.cpU), cpL: Array.from(r.linear.cpL) } : undefined,
  solver: r.solver ? { converged: r.solver.converged, sweeps: r.solver.totalSweeps, ms: r.solver.ms } : undefined,
};
if (args.json) console.log(JSON.stringify(out));
else console.log(`${out.engine}: valid=${out.valid} CL=${out.cl?.toFixed(5)} CD=${out.cd?.toFixed(6)} CM=${out.cm?.toFixed(5)}${out.warnings?.length ? '\n  ' + out.warnings.join('\n  ') : ''}`);
