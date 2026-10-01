// Web Worker: all flow solves and the optimizer run here so the page stays
// responsive (a cold TSD solve takes a second or more).
import { Engines, ENGINE_INFO } from './engines.js';
import { Optimizer } from './optimizer.js';

let engines = null;
let stopFlag = false;
let busy = Promise.resolve();

async function boot() {
  let model = null;
  try {
    const res = await fetch(new URL('../models/surrogate.json', import.meta.url));
    if (res.ok) model = await res.json();
  } catch { /* surrogate optional */ }
  engines = new Engines({ surrogateModel: model });
  postMessage({
    type: 'ready',
    surrogate: model ? { domain: model.meta.domain, source: model.meta.source, validation: model.meta.validation, architecture: model.meta.architecture, nTrain: model.meta.n_train } : null,
  });
}

function slim(r) {
  // geometry is recomputed on the main thread; drop it to keep messages small
  const { geometry, ...rest } = r;
  void geometry;
  return rest;
}

/** Secant iteration on alpha so the fixed shape meets the lift target. */
function trim(evaluate, shape, alpha0, clTarget) {
  let a0 = alpha0, r0 = evaluate(shape, a0);
  if (!r0.valid) return null;
  let a1 = a0 + 0.5, r1 = evaluate(shape, a1);
  for (let k = 0; k < 10 && r1.valid; k++) {
    if (Math.abs(r1.cl - clTarget) < 2e-4) break;
    const slope = (r1.cl - r0.cl) / (a1 - a0);
    if (!Number.isFinite(slope) || Math.abs(slope) < 1e-6) break;
    const a2 = Math.max(-8, Math.min(12, a1 + (clTarget - r1.cl) / slope));
    a0 = a1; r0 = r1;
    a1 = a2; r1 = evaluate(shape, a1);
  }
  if (!r1.valid || Math.abs(r1.cl - clTarget) > 5e-3) return null;
  return { alpha: a1, cd: r1.cd, cl: r1.cl };
}

async function optimize(msg) {
  const { id, engine, M, shape, alpha, clTarget, constraint, limit, maxIter } = msg;
  stopFlag = false;
  const gradientMode = ENGINE_INFO[engine].gradient;
  engines.clearAnchor();
  if (engine === 'tsd') engines.tsdOpt.reset();
  const opt = new Optimizer({
    evaluate: engines.optimizerEvaluate(engine, M),
    onAccept: engine === 'tsd' ? () => engines.setAnchor() : undefined,
    shape, alpha, clTarget, constraint, limit, gradientMode,
    fdStep: engine === 'tsd' ? 5e-4 : 2e-4,
    maxIter: maxIter ?? (engine === 'tsd' ? 30 : 80),
  });
  const t0 = performance.now();
  // Fair baseline: the starting shape trimmed (alpha only) to the target lift.
  let baseline = null;
  if (clTarget !== null && clTarget !== undefined) {
    baseline = trim(engines.optimizerEvaluate(engine, M), shape, alpha, clTarget);
    if (baseline) {
      opt.z[opt.z.length - 1] = baseline.alpha;
      if (engine === 'tsd') engines.clearAnchor();
      postMessage({ type: 'opt-baseline', id, baseline });
    }
  }
  try {
    const rec = opt.init();
    postMessage({ type: 'opt-progress', id, rec, result: slim(opt.r), ms: performance.now() - t0 });
  } catch (e) {
    postMessage({ type: 'opt-error', id, message: e.message });
    return;
  }
  const minStepMs = msg.minStepMs ?? 0;
  let computeMs = performance.now() - t0; // solver time only, excluding the animation pacing
  let lastStep = performance.now();
  while (!opt.done && !stopFlag) {
    // pace fast engines so the reshaping is watchable; also lets 'stop' in
    const wait = Math.max(0, minStepMs - (performance.now() - lastStep));
    await new Promise((r) => setTimeout(r, wait));
    lastStep = performance.now();
    if (stopFlag) break;
    const rec = opt.step();
    computeMs += performance.now() - lastStep;
    if (!rec) break;
    postMessage({ type: 'opt-progress', id, rec, result: slim(opt.r), ms: computeMs });
  }
  // A surrogate's optimum is only a claim until the physics model confirms it.
  let verification = null;
  if (engine === 'surrogate') {
    const z = opt.z;
    const truth = engines.evaluate('auto', z.slice(0, z.length - 1), M, z[z.length - 1], { withLinear: false });
    let base = null;
    if (baseline) base = engines.evaluate('auto', shape, M, baseline.alpha, { withLinear: false });
    verification = {
      engine: truth.engine, valid: truth.valid, cd: truth.cd, cl: truth.cl,
      baseCd: base?.valid ? base.cd : null, baseCl: base?.valid ? base.cl : null,
    };
  }
  postMessage({ type: 'opt-done', id, stopped: stopFlag, iterations: opt.iter, evals: opt.evals, ms: computeMs, verification });
}

onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'stop') { stopFlag = true; return; }
  busy = busy.then(async () => {
    if (!engines) await ready;
    if (msg.type === 'evaluate') {
      const t0 = performance.now();
      try {
        const r = engines.evaluate(msg.engine, msg.shape, msg.M, msg.alpha, { withLinear: true });
        postMessage({ type: 'result', id: msg.id, result: slim(r), ms: performance.now() - t0 });
      } catch (err) {
        postMessage({ type: 'result', id: msg.id, result: { valid: false, warnings: [err.message] }, ms: 0 });
      }
    } else if (msg.type === 'optimize') {
      await optimize(msg);
    }
  });
};

const ready = boot();
