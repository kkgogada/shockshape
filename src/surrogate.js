// Neural surrogate: a small fully connected network (tanh hidden layers,
// linear output) loaded from JSON. Forward pass plus exact reverse-mode
// gradients of any output with respect to the inputs, so the optimizer
// gets dCd/d(shape) and dCl/d(shape) for the price of one backward pass,
// independent of the number of design variables. That is the role the
// adjoint solve plays in SU2 shape optimization.
//
// Model JSON (written by pipeline/train_surrogate.py):
// {
//   "meta": { "inputs": [...12 names], "outputs": [...names],
//             "x_mean", "x_std", "y_mean", "y_std", "cp_stations": [...],
//             "domain": { "mach": [lo, hi], "alpha": [lo, hi] },
//             "source": "analytic-placeholder" | "su2-euler", ... },
//   "layers": [ { "W": [[out x in]], "b": [out], "act": "tanh" | "linear" }, ... ]
// }

export class Surrogate {
  constructor(model) {
    this.meta = model.meta;
    this.layers = model.layers.map((L) => ({
      W: L.W.map((row) => Float64Array.from(row)),
      b: Float64Array.from(L.b),
      act: L.act,
    }));
    this.nIn = this.layers[0].W[0].length;
    this.nOut = this.layers[this.layers.length - 1].b.length;
    const m = this.meta;
    this.xMean = Float64Array.from(m.x_mean);
    this.xStd = Float64Array.from(m.x_std);
    this.yMean = Float64Array.from(m.y_mean);
    this.yStd = Float64Array.from(m.y_std);
    this.idx = Object.fromEntries(m.outputs.map((n, k) => [n, k]));
  }

  /** Forward pass; keeps activations for a subsequent backward pass. */
  forward(input) {
    let a = new Float64Array(this.nIn);
    for (let k = 0; k < this.nIn; k++) a[k] = (input[k] - this.xMean[k]) / this.xStd[k];
    const acts = [a];
    for (const L of this.layers) {
      const z = new Float64Array(L.b.length);
      for (let o = 0; o < z.length; o++) {
        let s = L.b[o];
        const w = L.W[o];
        for (let i = 0; i < a.length; i++) s += w[i] * a[i];
        z[o] = L.act === 'tanh' ? Math.tanh(s) : s;
      }
      a = z;
      acts.push(a);
    }
    const y = new Float64Array(this.nOut);
    for (let k = 0; k < this.nOut; k++) y[k] = a[k] * this.yStd[k] + this.yMean[k];
    return { y, acts };
  }

  /**
   * Gradient of sum_k seed[k] * y[k] with respect to the (unnormalised) inputs.
   * One reverse sweep through the network.
   */
  backward(acts, seed) {
    let g = new Float64Array(this.nOut);
    for (let k = 0; k < this.nOut; k++) g[k] = seed[k] * this.yStd[k];
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const L = this.layers[l];
      const out = acts[l + 1], inp = acts[l];
      if (L.act === 'tanh') for (let o = 0; o < g.length; o++) g[o] *= 1 - out[o] * out[o];
      const gi = new Float64Array(inp.length);
      for (let o = 0; o < g.length; o++) {
        const w = L.W[o], go = g[o];
        if (go === 0) continue;
        for (let i = 0; i < inp.length; i++) gi[i] += w[i] * go;
      }
      g = gi;
    }
    for (let k = 0; k < this.nIn; k++) g[k] /= this.xStd[k];
    return g;
  }

  inDomain(M, alpha) {
    const d = this.meta.domain;
    return M >= d.mach[0] && M <= d.mach[1] && alpha >= d.alpha[0] && alpha <= d.alpha[1];
  }

  /** Evaluate in the common engine format, optionally with exact gradients. */
  evaluate(shape, M, alphaDeg, { gradients = false } = {}) {
    const input = [...shape, M, alphaDeg];
    const { y, acts } = this.forward(input);
    const nS = this.meta.cp_stations.length;
    const iU = this.idx['cpU_0'], iL = this.idx['cpL_0'];
    const out = {
      engine: 'surrogate',
      valid: true,
      warnings: [],
      x: Float64Array.from(this.meta.cp_stations),
      cpU: y.slice(iU, iU + nS),
      cpL: y.slice(iL, iL + nS),
      cl: y[this.idx.cl],
      cd: y[this.idx.cd],
      cm: this.idx.cm !== undefined ? y[this.idx.cm] : NaN,
    };
    if (!this.inDomain(M, alphaDeg)) {
      out.warnings.push(`Outside the surrogate's training range (M ${this.meta.domain.mach.join('–')}, α ${this.meta.domain.alpha.join('–')}°): this is extrapolation.`);
    }
    if (gradients) {
      const seed = new Float64Array(this.nOut);
      seed[this.idx.cd] = 1;
      const gCd = this.backward(acts, seed);
      seed.fill(0);
      seed[this.idx.cl] = 1;
      const gCl = this.backward(acts, seed);
      // inputs are [shape..., M, alpha]; the optimizer wants d/d(shape, alpha)
      const pick = (g) => Float64Array.from([...g.slice(0, shape.length), g[shape.length + 1]]);
      out.grad = { cd: pick(gCd), cl: pick(gCl) };
    }
    return out;
  }
}
