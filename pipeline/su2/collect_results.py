"""Gather finished SU2 cases into the surrogate training CSV.

Reads each case's design.csv, the last row of history.csv (CL, CD, CMz) and
surface_flow.csv (x, y, Pressure_Coefficient). Surface points are split into
upper and lower surfaces against the design's camber line and Cp is
interpolated onto the app's 32 fixed stations, so the output has exactly the
same columns as tools/generate-analytic-dataset.mjs:

    h0..h9, mach, alpha, cl, cd, cm, cpU_0..cpU_31, cpL_0..cpL_31

Cases whose density residual did not drop by --min-drop orders are skipped
and listed, rather than silently polluting the training set.

Usage:
    python pipeline/su2/collect_results.py cases/ data/dataset_su2.csv
then
    python pipeline/train_surrogate.py data/dataset_su2.csv models/surrogate.json --source su2-euler
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry import surfaces  # noqa: E402

STATIONS = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "..", "cp_stations.json"))))


def read_history(path):
    with open(path) as f:
        r = csv.reader(f)
        header = [h.strip().strip('"') for h in next(r)]
        rows = [[float(v) for v in row] for row in r if row]
    first, last = dict(zip(header, rows[0])), dict(zip(header, rows[-1]))
    return first, last


def read_surface(path):
    with open(path) as f:
        r = csv.reader(f)
        header = [h.strip().strip('"') for h in next(r)]
        data = np.array([[float(v) for v in row] for row in r if row])
    col = {h: k for k, h in enumerate(header)}
    return data[:, col["x"]], data[:, col["y"]], data[:, col["Pressure_Coefficient"]]


def split_and_resample(shape, x, y, cp):
    yu, yl = surfaces(shape, np.clip(x, 0, 1))
    camber = 0.5 * (yu + yl)
    out = []
    for upper in (True, False):
        m = (y >= camber) if upper else (y < camber)
        xs, cs = x[m], cp[m]
        order = np.argsort(xs)
        out.append(np.interp(STATIONS, xs[order], cs[order]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases_dir")
    p.add_argument("out")
    p.add_argument("--min-drop", type=float, default=3.0, help="required drop in log10 rms[Rho]")
    a = p.parse_args()

    rows, skipped = [], []
    for d in sorted(glob.glob(os.path.join(a.cases_dir, "case_*"))):
        try:
            with open(os.path.join(d, "design.csv")) as f:
                design = next(csv.DictReader(f))
            first, last = read_history(os.path.join(d, "history.csv"))
            drop = first["rms[Rho]"] - last["rms[Rho]"]
            if drop < a.min_drop:
                skipped.append((d, f"residual dropped only {drop:.1f} orders"))
                continue
            shape = [float(design[f"h{j}"]) for j in range(10)]
            cpU, cpL = split_and_resample(shape, *read_surface(os.path.join(d, "surface_flow.csv")))
            rows.append([*shape, float(design["mach"]), float(design["alpha"]),
                         last["CL"], last["CD"], last["CMz"], *cpU, *cpL])
        except (OSError, KeyError, StopIteration, ValueError) as e:
            skipped.append((d, f"{type(e).__name__}: {e}"))
    header = ([f"h{k}" for k in range(10)] + ["mach", "alpha", "cl", "cd", "cm"]
              + [f"cpU_{k}" for k in range(len(STATIONS))] + [f"cpL_{k}" for k in range(len(STATIONS))])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows([[f"{v:.8g}" for v in r] for r in rows])
    print(f"wrote {len(rows)} cases to {a.out}")
    for d, why in skipped:
        print(f"  skipped {d}: {why}")


if __name__ == "__main__":
    main()
