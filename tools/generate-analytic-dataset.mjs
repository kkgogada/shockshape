// Placeholder training data: evaluates every design point with shock-expansion
// theory and writes the same CSV layout that pipeline/su2/collect_results.py
// produces from SU2 runs. Lets the whole surrogate chain be built and tested
// before the SU2 sweep exists.
//
//   node tools/generate-analytic-dataset.mjs data/design_points.csv data/dataset.csv
import { readFileSync, writeFileSync } from 'node:fs';
import { evaluateShockExpansion } from '../src/shockExpansion.js';
import { CP_STATIONS, resampleCp } from '../src/stations.js';

const [, , inPath = 'data/design_points.csv', outPath = 'data/dataset.csv'] = process.argv;
const lines = readFileSync(inPath, 'utf8').trim().split(/\r?\n/);
const header = lines[0].split(',');
const outHeader = [
  ...header, 'cl', 'cd', 'cm',
  ...CP_STATIONS.map((_, k) => `cpU_${k}`), ...CP_STATIONS.map((_, k) => `cpL_${k}`),
];
const rows = [outHeader.join(',')];
let skipped = 0;
for (const line of lines.slice(1)) {
  const v = line.split(',').map(Number);
  const shape = v.slice(0, 10), M = v[10], alpha = v[11];
  const r = evaluateShockExpansion(shape, M, alpha, { withLinear: false });
  if (!r.valid) { skipped++; continue; }
  const cpU = resampleCp(r.x, r.cpU), cpL = resampleCp(r.x, r.cpL);
  rows.push([...v, r.cl, r.cd, r.cm, ...cpU, ...cpL].map((n) => +n.toPrecision(8)).join(','));
}
writeFileSync(outPath, rows.join('\n') + '\n');
console.log(`wrote ${rows.length - 1} samples to ${outPath} (${skipped} skipped: detached shock / invalid)`);
