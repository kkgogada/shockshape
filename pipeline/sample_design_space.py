"""Latin-hypercube sample of the design space: 10 handle heights + Mach + alpha.

Shapes are sampled as a half-thickness and a camber distribution, each a
parabolic base shape (amplitude T or C) with independent +/- perturbations at
every handle station. This keeps samples closed and non-crossing, with
realistic leading-edge angles, while still covering diamond-like, aft- and
fore-loaded shapes:
    tau_j = T * 4 x_j (1 - x_j) * (1 + eps_j)
    cam_j = C * 4 x_j (1 - x_j) * (1 + del_j)
    upper = cam + tau,   lower = cam - tau

Usage:
    python pipeline/sample_design_space.py --n 6000 --mach 1.4 3.0 --alpha -2 6 \
        --out data/design_points.csv
"""
import argparse
import csv
import os

import numpy as np

from geometry import HANDLE_X, N_HANDLES


def latin_hypercube(n, d, rng):
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        cut[:, j] = cut[rng.permutation(n), j]
    return cut


def sample(n, mach, alpha, half_thickness, camber, perturb=0.35, seed=0):
    rng = np.random.default_rng(seed)
    u = latin_hypercube(n, 2 * N_HANDLES + 4, rng)
    base = 4 * HANDLE_X * (1 - HANDLE_X)
    T = half_thickness[0] + u[:, 0] * (half_thickness[1] - half_thickness[0])
    C = camber[0] + u[:, 1] * (camber[1] - camber[0])
    eps = (2 * u[:, 2:2 + N_HANDLES] - 1) * perturb
    dlt = (2 * u[:, 2 + N_HANDLES:2 + 2 * N_HANDLES] - 1) * perturb
    tau = T[:, None] * base * (1 + eps)
    cam = C[:, None] * base * (1 + dlt)
    m = mach[0] + u[:, -2] * (mach[1] - mach[0])
    a = alpha[0] + u[:, -1] * (alpha[1] - alpha[0])
    upper, lower = cam + tau, cam - tau
    return np.column_stack([upper, lower, m, a])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--mach", type=float, nargs=2, default=[1.4, 3.0])
    p.add_argument("--alpha", type=float, nargs=2, default=[-2.0, 6.0])
    p.add_argument("--half-thickness", type=float, nargs=2, default=[0.01, 0.04], help="half t/c amplitude range")
    p.add_argument("--camber", type=float, nargs=2, default=[-0.015, 0.015])
    p.add_argument("--perturb", type=float, default=0.35, help="relative per-station perturbation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/design_points.csv")
    a = p.parse_args()
    pts = sample(a.n, a.mach, a.alpha, a.half_thickness, a.camber, a.perturb, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    header = [f"h{k}" for k in range(2 * N_HANDLES)] + ["mach", "alpha"]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(pts.tolist())
    print(f"wrote {len(pts)} design points to {a.out}")


if __name__ == "__main__":
    main()
