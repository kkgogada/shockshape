"""Plot a finished SU2 case against the app's physics models (surface Cp).

    python pipeline/su2/compare_case.py cases/case_00000 docs/validation.png [--engines shock-expansion tsd]

Reads the case's design.csv and surface_flow.csv, evaluates the same design
with tools/evaluate.mjs (Node) and overlays the surface pressures.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry import surfaces  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def app_eval(shape, mach, alpha, engine):
    out = subprocess.run(
        ["node", "tools/evaluate.mjs", "--engine", engine, "--mach", str(mach), "--alpha", str(alpha),
         "--shape", ",".join(f"{v:.10g}" for v in shape), "--json"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("out")
    p.add_argument("--engines", nargs="+", default=["auto"])
    p.add_argument("--title", default=None)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    design = next(csv.DictReader(open(os.path.join(a.case, "design.csv"))))
    shape = [float(design[f"h{j}"]) for j in range(10)]
    mach, alpha = float(design["mach"]), float(design["alpha"])
    with open(os.path.join(a.case, "surface_flow.csv")) as f:
        r = csv.reader(f)
        hdr = [h.strip().strip('"') for h in next(r)]
        d = np.array([[float(v) for v in row] for row in r if row])
    x, y, cp = d[:, hdr.index("x")], d[:, hdr.index("y")], d[:, hdr.index("Pressure_Coefficient")]
    with open(os.path.join(a.case, "history.csv")) as f:
        r = csv.reader(f)
        hh = [h.strip().strip('"') for h in next(r)]
        last = [row for row in r if row][-1]
    su2_cl, su2_cd = float(last[hh.index("CL")]), float(last[hh.index("CD")])
    yu, yl = surfaces(shape, np.clip(x, 0, 1))
    up = y >= 0.5 * (yu + yl)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888", "axes.labelcolor": "#222"})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    o = np.argsort(x[up]); ax.plot(x[up][o], cp[up][o], color="#111", lw=2, label=f"SU2 Euler upper  (CL {su2_cl:.4f}, CD {su2_cd:.5f})")
    o = np.argsort(x[~up]); ax.plot(x[~up][o], cp[~up][o], color="#111", lw=2, ls=(0, (1, 1.5)), label="SU2 Euler lower")
    colors = {"shock-expansion": "#e5383b", "tsd": "#2f6db3", "surrogate": "#7a4cc2"}
    for eng in a.engines:
        res = app_eval(shape, mach, alpha, eng)
        c = colors.get(res["engine"], "#e5383b")
        name = {"shock-expansion": "Shock-expansion", "tsd": "TSD", "surrogate": "Surrogate"}[res["engine"]]
        ax.plot(res["x"], res["cpU"], color=c, lw=1.4, label=f"{name} upper  (CL {res['cl']:.4f}, CD {res['cd']:.5f})")
        ax.plot(res["x"], res["cpL"], color=c, lw=1.4, ls="--", label=f"{name} lower")
        if res.get("linear") and eng == "shock-expansion":
            ax.plot(res["x"], res["linear"]["cpU"], color="#999", lw=1, label=f"Ackeret (CL {res['linear']['cl']:.4f}, CD {res['linear']['cd']:.5f})")
            ax.plot(res["x"], res["linear"]["cpL"], color="#999", lw=1, ls="--")
    # SU2's sharp-edge nodes carry single-point spikes; frame the plot on the app curves
    lo = min(min(res["cpU"]), min(res["cpL"])); hi = max(max(res["cpU"]), max(res["cpL"]))
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.invert_yaxis()
    ax.set_xlabel("x/c"); ax.set_ylabel("Cp")
    ax.set_title(a.title or f"M = {mach:g}, α = {alpha:g}°")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}: SU2 CL {su2_cl:.5f} CD {su2_cd:.6f}")


if __name__ == "__main__":
    main()
