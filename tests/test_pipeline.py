"""Pipeline checks: Python geometry == JS geometry, sampler sanity, SU2 result parsing.

Run: python -m unittest discover -s tests -p "test_*.py"
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline", "su2"))

import geometry  # noqa: E402
from sample_design_space import sample  # noqa: E402


class GeometryParity(unittest.TestCase):
    def test_matches_js(self):
        cases = json.load(open(os.path.join(HERE, "fixtures", "geometry.json")))
        for c in cases:
            yu, yl = geometry.surfaces(c["shape"], np.array(c["x"]))
            np.testing.assert_allclose(yu, c["yu"], atol=1e-12, err_msg=c["name"])
            np.testing.assert_allclose(yl, c["yl"], atol=1e-12, err_msg=c["name"])

    def test_coordinates_closed_selig_order(self):
        xy = geometry.coordinates([0.01] * 5 + [-0.01] * 5, n=51)
        self.assertEqual(tuple(xy[0]), (1.0, 0.0))
        self.assertEqual(tuple(xy[-1]), (1.0, 0.0))
        self.assertAlmostEqual(xy[50][0], 0.0)
        self.assertTrue(np.all(xy[1:50, 1] > 0) and np.all(xy[51:-1, 1] < 0))


class Sampler(unittest.TestCase):
    def test_ranges_and_no_crossing(self):
        pts = sample(500, [1.4, 3.0], [-2, 6], [0.01, 0.04], [-0.015, 0.015], 0.35, seed=1)
        self.assertEqual(pts.shape, (500, 12))
        self.assertTrue(np.all(pts[:, :5] > pts[:, 5:10]))  # upper above lower at every handle
        self.assertTrue(pts[:, 10].min() >= 1.4 and pts[:, 10].max() <= 3.0)
        # Latin hypercube: each Mach decile holds ~10% of the samples
        counts = np.histogram(pts[:, 10], bins=10, range=(1.4, 3.0))[0]
        self.assertTrue(np.all(np.abs(counts - 50) <= 1))


class CollectResults(unittest.TestCase):
    def test_parses_su2_output_format(self):
        from collect_results import main as collect
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "case_00000")
            os.makedirs(d)
            shape = [0.12 * x * (1 - x) for x in geometry.HANDLE_X] + [-0.12 * x * (1 - x) for x in geometry.HANDLE_X]
            with open(os.path.join(d, "design.csv"), "w") as f:
                f.write(",".join([f"h{k}" for k in range(10)] + ["mach", "alpha"]) + "\n")
                f.write(",".join(map(str, shape + [2.0, 0.0])) + "\n")
            with open(os.path.join(d, "history.csv"), "w") as f:
                f.write('"Inner_Iter",    "rms[Rho]"    ,       "CD"       ,       "CL"       ,      "CMz"\n')
                f.write("0, -1.0, 0.02, 0.0, 0.0\n")
                f.write("900, -7.5, 0.0141, 0.0001, 0.0002\n")
            x = geometry.cosine_stations(101)
            yu, yl = geometry.surfaces(shape, x)
            with open(os.path.join(d, "surface_flow.csv"), "w") as f:
                f.write('"PointID","x","y","Pressure_Coefficient"\n')
                k = 0
                for xx, yy, cp in [*zip(x, yu, 0.1 - 0.2 * x), *zip(x[1:-1], yl[1:-1], 0.3 - 0.2 * x[1:-1])]:
                    f.write(f"{k},{xx},{yy},{cp}\n")
                    k += 1
            out = os.path.join(tmp, "ds.csv")
            sys.argv = ["collect", tmp, out]
            collect()
            with open(out) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertAlmostEqual(float(r["cd"]), 0.0141)
            st = json.load(open(os.path.join(HERE, "..", "pipeline", "cp_stations.json")))
            self.assertAlmostEqual(float(r["cpU_5"]), 0.1 - 0.2 * st[5], places=3)
            self.assertAlmostEqual(float(r["cpL_20"]), 0.3 - 0.2 * st[20], places=3)


@unittest.skipUnless(subprocess.run(["which", "node"], capture_output=True).returncode == 0, "node not installed")
class EndToEnd(unittest.TestCase):
    def test_sample_dataset_train_export(self):
        root = os.path.join(HERE, "..")
        with tempfile.TemporaryDirectory() as tmp:
            pts, ds, model = (os.path.join(tmp, n) for n in ("p.csv", "d.csv", "m.json"))
            subprocess.run([sys.executable, "pipeline/sample_design_space.py", "--n", "300", "--out", pts], cwd=root, check=True, capture_output=True)
            subprocess.run(["node", "tools/generate-analytic-dataset.mjs", pts, ds], cwd=root, check=True, capture_output=True)
            subprocess.run([sys.executable, "pipeline/train_surrogate.py", ds, model, "--epochs", "30", "--hidden", "16"], cwd=root, check=True, capture_output=True)
            m = json.load(open(model))
            self.assertEqual(m["meta"]["inputs"][-2:], ["mach", "alpha"])
            self.assertEqual(len(m["layers"]), 2)
            self.assertEqual(len(m["layers"][-1]["b"]), 3 + 64)


if __name__ == "__main__":
    unittest.main()
