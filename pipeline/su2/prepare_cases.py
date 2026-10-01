"""Turn design points into ready-to-run SU2 Euler cases.

For every row of design_points.csv this writes cases/case_NNNNN/ containing
    airfoil.dat   coordinates (Selig order), from the same parameterization as the app
    mesh.su2      unstructured triangle mesh (gmsh), markers 'airfoil' and 'farfield'
    euler.cfg     SU2 configuration (Mach and AoA filled in)
and a run_all.sh that runs them (optionally in parallel with GNU parallel / xargs).

Euler rather than RANS on purpose: wave drag dominates the supersonic story,
and inviscid runs are cheap enough to generate a training set at volume.

Usage:
    python pipeline/su2/prepare_cases.py data/design_points.csv cases/ [--limit 50]
Requires: pip install gmsh numpy
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry import coordinates  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(__file__), "euler_template.cfg")


def write_mesh(coords, path, farfield_radius=25.0, h_wall=0.004, h_far=2.5, h_wake=0.03):
    import gmsh

    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", int(os.environ.get("GMSH_VERBOSE", "0")))
    gmsh.model.add("airfoil")
    occ = gmsh.model.geo
    # airfoil loop: coords are TE -> upper -> LE -> lower -> TE (closed at TE)
    pts = [occ.addPoint(x, y, 0, h_wall) for x, y in coords[:-1]]
    spline_up = occ.addSpline(pts[: len(pts) // 2 + 1])
    spline_lo = occ.addSpline(pts[len(pts) // 2:] + [pts[0]])
    loop_af = occ.addCurveLoop([spline_up, spline_lo])
    c = occ.addPoint(0.5, 0, 0, h_far)
    far = [occ.addPoint(0.5 + farfield_radius * np.cos(t), farfield_radius * np.sin(t), 0, h_far)
           for t in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    arcs = [occ.addCircleArc(far[k], c, far[(k + 1) % 4]) for k in range(4)]
    loop_far = occ.addCurveLoop(arcs)
    surf = occ.addPlaneSurface([loop_far, loop_af])
    occ.synchronize()
    # refinement box around the airfoil and its shock / wave system
    f = gmsh.model.mesh.field
    box = f.add("Box")
    f.setNumber(box, "VIn", h_wake)
    f.setNumber(box, "VOut", h_far)
    f.setNumber(box, "XMin", -0.6)
    f.setNumber(box, "XMax", 2.5)
    f.setNumber(box, "YMin", -1.2)
    f.setNumber(box, "YMax", 1.2)
    dist = f.add("Distance")
    f.setNumbers(dist, "CurvesList", [spline_up, spline_lo])
    th = f.add("Threshold")
    f.setNumber(th, "InField", dist)
    f.setNumber(th, "SizeMin", h_wall)
    f.setNumber(th, "SizeMax", h_far)  # beyond DistMax the box / far-field size governs
    f.setNumber(th, "DistMin", 0.02)
    f.setNumber(th, "DistMax", 0.6)
    f.setNumber(dist, "Sampling", 400)
    mn = f.add("Min")
    f.setNumbers(mn, "FieldsList", [box, th])
    f.setAsBackgroundMesh(mn)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.model.addPhysicalGroup(1, [spline_up, spline_lo], name="airfoil")
    gmsh.model.addPhysicalGroup(1, arcs, name="farfield")
    gmsh.model.addPhysicalGroup(2, [surf], name="fluid")
    gmsh.model.mesh.generate(2)
    n_el = sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])
    gmsh.write(path)
    gmsh.finalize()
    return n_el


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("design_points")
    p.add_argument("out_dir")
    p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    p.add_argument("--iters", type=int, default=4000)
    a = p.parse_args()

    template = open(TEMPLATE).read()
    with open(a.design_points) as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    names = []
    for k, row in enumerate(rows):
        shape = [float(row[f"h{j}"]) for j in range(10)]
        mach, alpha = float(row["mach"]), float(row["alpha"])
        d = os.path.join(a.out_dir, f"case_{k:05d}")
        os.makedirs(d, exist_ok=True)
        xy = coordinates(shape, n=241)
        np.savetxt(os.path.join(d, "airfoil.dat"), xy, fmt="%.8f", header="shockshape design", comments="")
        n_el = write_mesh(xy, os.path.join(d, "mesh.su2"))
        cfg = (template.replace("@MACH@", f"{mach:.6f}")
               .replace("@AOA@", f"{alpha:.6f}")
               .replace("@ITER@", str(a.iters)))
        open(os.path.join(d, "euler.cfg"), "w").write(cfg)
        with open(os.path.join(d, "design.csv"), "w") as f:
            f.write(",".join(row.keys()) + "\n" + ",".join(row.values()) + "\n")
        names.append(os.path.basename(d))
        print(f"{d}: M={mach:.3f} AoA={alpha:.2f} ({n_el} triangles)")
    with open(os.path.join(a.out_dir, "run_all.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\n# Runs every case; set JOBS for parallel runs (needs xargs -P).\n")
        f.write('cd "$(dirname "$0")"\nJOBS=${JOBS:-1}\n')
        f.write("ls -d case_* | xargs -P \"$JOBS\" -I{} sh -c 'cd {} && SU2_CFD euler.cfg > log.txt 2>&1 && echo done {}'\n")
    os.chmod(os.path.join(a.out_dir, "run_all.sh"), 0o755)
    print(f"prepared {len(names)} cases; run {a.out_dir}/run_all.sh, then collect_results.py")


if __name__ == "__main__":
    main()
