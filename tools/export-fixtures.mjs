// Writes JS-computed geometry so the Python pipeline can be checked against it.
import { writeFileSync } from 'node:fs';
import { surfaces, PRESETS, cosineStations } from '../src/geometry.js';

const xs = cosineStations(41);
const cases = Object.entries(PRESETS).map(([name, shape]) => {
  const s = surfaces(shape, xs);
  return { name, shape, x: Array.from(xs), yu: Array.from(s.yu), yl: Array.from(s.yl) };
});
writeFileSync(new URL('../tests/fixtures/geometry.json', import.meta.url), JSON.stringify(cases));
console.log(`wrote ${cases.length} geometry fixtures`);
