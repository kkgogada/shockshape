"""Train the neural surrogate and export it for the browser.

Input  : 10 handle heights + Mach + alpha (deg)
Output : Cl, Cd, Cm and Cp at 32 stations on each surface

A small MLP with tanh hidden layers. tanh rather than ReLU on purpose: the
optimizer uses gradients of the network, and a ReLU net's gradient is
piecewise constant, which makes a poor stand-in for the smooth adjoint
sensitivities it replaces.

Pure NumPy (no framework) so the exported weights map one-to-one onto the
~80-line forward/backward pass in src/surrogate.js.

Usage:
    python pipeline/train_surrogate.py data/dataset.csv models/surrogate.json \
        --source analytic-placeholder --epochs 1500
"""
import argparse
import csv
import json
import time

import numpy as np

INPUTS = [f"h{k}" for k in range(10)] + ["mach", "alpha"]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        data = np.array([[float(v) for v in row] for row in r])
    outputs = [h for h in header if h not in INPUTS]
    X = data[:, [header.index(h) for h in INPUTS]]
    Y = data[:, [header.index(h) for h in outputs]]
    return X, Y, outputs


def init(sizes, rng):
    return [
        {"W": rng.normal(0, np.sqrt(1.0 / a), (b, a)), "b": np.zeros(b)}
        for a, b in zip(sizes[:-1], sizes[1:])
    ]


def forward(params, X):
    acts = [X]
    a = X
    for k, p in enumerate(params):
        z = a @ p["W"].T + p["b"]
        a = np.tanh(z) if k < len(params) - 1 else z
        acts.append(a)
    return acts


def train(Xn, Yn, w_out, sizes, epochs, batch, lr, rng, Xv, Yv, log_every=100):
    params = init(sizes, rng)
    m = [{k: np.zeros_like(v) for k, v in p.items()} for p in params]
    v = [{k: np.zeros_like(v) for k, v in p.items()} for p in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best, best_params = np.inf, None
    n = len(Xn)
    for ep in range(epochs):
        lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs)) + 1e-5  # cosine decay
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            acts = forward(params, Xn[idx])
            g = 2 * (acts[-1] - Yn[idx]) * w_out / len(idx)
            for k in range(len(params) - 1, -1, -1):
                gW = g.T @ acts[k]
                gb = g.sum(0)
                if k > 0:
                    g = (g @ params[k]["W"]) * (1 - acts[k] ** 2)
                t += 1
                for name, grad in (("W", gW), ("b", gb)):
                    m[k][name] = b1 * m[k][name] + (1 - b1) * grad
                    v[k][name] = b2 * v[k][name] + (1 - b2) * grad**2
                    mh = m[k][name] / (1 - b1**t)
                    vh = v[k][name] / (1 - b2**t)
                    params[k][name] -= lr_ep * mh / (np.sqrt(vh) + eps)
        val = float(np.mean((forward(params, Xv)[-1] - Yv) ** 2 * w_out))
        if val < best:
            best = val
            best_params = [{k: x.copy() for k, x in p.items()} for p in params]
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  epoch {ep:5d}  val loss {val:.3e}")
    return best_params


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("out")
    p.add_argument("--source", default="analytic-placeholder",
                   help="provenance label shown in the app (e.g. su2-euler)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    X, Y, outputs = load(a.dataset)
    perm = rng.permutation(len(X))
    nv = int(len(X) * a.val_frac)
    iv, it = perm[:nv], perm[nv:]
    xm, xs = X[it].mean(0), X[it].std(0) + 1e-12
    ym, ys = Y[it].mean(0), Y[it].std(0) + 1e-12
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    # the integrated coefficients drive the optimizer: weight them up
    w_out = np.ones(len(outputs))
    for name in ("cl", "cd"):
        w_out[outputs.index(name)] = 8.0

    sizes = [X.shape[1], *a.hidden, Y.shape[1]]
    print(f"training {sizes} on {len(it)} samples, validating on {nv}")
    t0 = time.time()
    params = train(Xn[it], Yn[it], w_out, sizes, a.epochs, a.batch, a.lr, rng, Xn[iv], Yn[iv])
    print(f"done in {time.time() - t0:.0f}s")

    pred = forward(params, Xn[iv])[-1] * ys + ym
    metrics = {}
    for name in ("cl", "cd", "cm"):
        k = outputs.index(name)
        err = pred[:, k] - Y[iv, k]
        ss = np.sum((Y[iv, k] - Y[iv, k].mean()) ** 2)
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs": float(np.max(np.abs(err))),
            "r2": float(1 - np.sum(err**2) / ss),
        }
    cpk = [k for k, o in enumerate(outputs) if o.startswith("cp")]
    metrics["cp_rmse"] = float(np.sqrt(np.mean((pred[:, cpk] - Y[iv][:, cpk]) ** 2)))
    print(json.dumps(metrics, indent=2))

    from_stations = json.load(open(__file__.replace("train_surrogate.py", "cp_stations.json")))
    model = {
        "meta": {
            "inputs": INPUTS,
            "outputs": outputs,
            "x_mean": xm.tolist(), "x_std": xs.tolist(),
            "y_mean": ym.tolist(), "y_std": ys.tolist(),
            "cp_stations": from_stations,
            "domain": {
                "mach": [float(X[:, 10].min()), float(X[:, 10].max())],
                "alpha": [float(X[:, 11].min()), float(X[:, 11].max())],
            },
            "source": a.source,
            "n_train": int(len(it)),
            "n_val": int(nv),
            "architecture": sizes,
            "validation": metrics,
        },
        "layers": [
            {"W": np.round(p_["W"], 7).tolist(), "b": np.round(p_["b"], 7).tolist(),
             "act": "tanh" if k < len(params) - 1 else "linear"}
            for k, p_ in enumerate(params)
        ],
    }
    with open(a.out, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
