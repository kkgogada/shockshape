// Engine registry: every physics model answers the same question,
//   (shape, Mach, alpha) -> { cpU, cpL, cl, cd, valid, warnings, ... }
// so the front end and the optimizer never care which one is running.
// The trained SU2 surrogate drops in here without touching anything else.

import { evaluateShockExpansion } from './shockExpansion.js';
import { TSDSolver } from './tsd.js';
import { Surrogate } from './surrogate.js';

export const SUPERSONIC_SWITCH = 1.3; // auto: shock-expansion at and above this Mach

export const ENGINE_INFO = {
  'shock-expansion': {
    label: 'Shock-expansion theory',
    gradient: 'central',
    blurb: 'Oblique shock or Prandtl–Meyer expansion at the leading edge, then isentropic simple waves along each surface. Exact inviscid surface solution for sharp airfoils with an attached shock.',
  },
  tsd: {
    label: 'Transonic small-disturbance solver',
    gradient: 'forward',
    blurb: 'Nonlinear TSD equation solved on a 2-D grid (Murman–Cole type-dependent differencing, line relaxation). Captures embedded shocks, the sonic line and wave drag through Mach 1.',
  },
  surrogate: {
    label: 'Neural surrogate',
    gradient: 'exact',
    blurb: 'Small MLP trained offline on a sweep of flow solutions. Millisecond evaluation and exact gradients by back-propagation: the adjoint’s role in the design loop.',
  },
};

export class Engines {
  constructor({ surrogateModel = null, tsdGrid = 40, tsdOptGrid = 32 } = {}) {
    this.tsdView = new TSDSolver({ na: tsdGrid });
    this.tsdOpt = new TSDSolver({ na: tsdOptGrid });
    this.surrogate = surrogateModel ? new Surrogate(surrogateModel) : null;
    this.anchor = null;
  }

  /** Which engine 'auto' resolves to. */
  resolve(name, shape, M, alpha) {
    if (name !== 'auto') return name;
    if (M >= SUPERSONIC_SWITCH) {
      const r = evaluateShockExpansion(shape, M, alpha, { withLinear: false });
      if (r.valid || M > 1.6) return 'shock-expansion';
    }
    return 'tsd';
  }

  evaluate(name, shape, M, alpha, opts = {}) {
    const which = this.resolve(name, shape, M, alpha);
    let r;
    if (which === 'shock-expansion') {
      r = evaluateShockExpansion(shape, M, alpha, { withLinear: opts.withLinear !== false });
      if (M < 1) {
        r.valid = false;
        r.warnings = ['Shock-expansion theory needs supersonic free-stream flow (M > 1).'];
      }
    } else if (which === 'tsd') {
      r = (opts.forOptimizer ? this.tsdOpt : this.tsdView).solve(shape, M, alpha, { timeBudgetMs: opts.forOptimizer ? 6000 : 9000, ...opts });
      if (!r.solver.converged && !(r.solver.corr < 1e-5)) {
        // an unconverged transonic iterate is not a flow solution; don't let
        // the optimizer (or the readouts) treat it as one
        r.valid = false;
        r.warnings.unshift('The transonic solver did not converge for this case, so no forces are reported. Strongly lifting or thick sections near Mach 1 are the hardest cases for this method; try a smaller angle of attack or a thinner section.');
      }
      const t = maxThicknessQuick(r.geometry);
      if (t > 0.1) r.warnings.push(`TSD assumes a thin airfoil; t/c = ${(t * 100).toFixed(1)}% is pushing it.`);
      if (M > 1.6) r.warnings.push('Above M ≈ 1.6 the small-disturbance scaling degrades; shock-expansion is the better model here.');
    } else if (which === 'surrogate') {
      if (!this.surrogate) return { valid: false, engine: 'surrogate', warnings: ['No surrogate model loaded.'] };
      r = this.surrogate.evaluate(shape, M, alpha, opts);
    } else {
      throw new Error(`unknown engine ${which}`);
    }
    r.engine = which;
    return r;
  }

  /** Engine call adapted for the optimizer (warm-started TSD with a fixed anchor). */
  optimizerEvaluate(name, M) {
    return (shape, alpha, { gradients } = {}) => {
      if (name === 'tsd') {
        if (this.anchor) this.restore(this.anchor);
        return this.evaluate('tsd', shape, M, alpha, { forOptimizer: true, tol: 2e-7 });
      }
      return this.evaluate(name, shape, M, alpha, { gradients, withLinear: false });
    };
  }

  snapshot() {
    const l = this.tsdOpt.fine;
    return { phi: l.phi.slice(), gamma: l.gamma, key: this.tsdOpt.warmKey };
  }

  restore(s) {
    const l = this.tsdOpt.fine;
    l.phi.set(s.phi);
    l.gamma = s.gamma;
    this.tsdOpt.warmKey = s.key;
  }

  /** Called when the optimizer accepts a design: perturbations warm-start from it. */
  setAnchor() {
    this.anchor = this.snapshot();
  }

  clearAnchor() {
    this.anchor = null;
  }
}

function maxThicknessQuick(geom) {
  let t = 0;
  for (let k = 0; k < geom.yu.length; k++) t = Math.max(t, geom.yu[k] - geom.yl[k]);
  return t;
}
