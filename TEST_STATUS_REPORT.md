---

# Automated Test & Validation Summary

| Metric | Result |
| --- | --- |
| Overall status | All automated tests passing |
| Total tests (pytest) | 130 |
| Test runtime | ~0.9 seconds |
| Linter / static analysis | No issues reported |

---

## Suite Composition

| Category | File(s) | # Tests | Description |
| --- | --- | ---:| --- |
| Integration regression | `tests/test_pa3.py` | 6 | End-to-end PA3 pipeline for datasets A–F with per-dataset tolerances |
| Expanded regression + file diffs | `tests/test_all_output_files.py` | 17 | Re-runs pipeline, compares against answer files, validates logfile, smoke-tests unknown datasets |
| Transforms | `tests/test_transforms.py` | 20 | Rigid registration, transform inversion, transform application, PA3-A validation |
| Geometry | `tests/test_geometry.py` | 21 | Triangle/mesh nearest-point logic, PA3-A validation, KD-tree parity check |
| I/O | `tests/test_io.py` | 18 | Mesh, rigid body, sample loading (happy paths + error cases) |
| Matching | `tests/test_matching.py` | 10 | `compute_matches` with mock and real datasets (A–C) |
| Output formatting | `tests/test_output.py` | 6 | Ensures numerical precision and string layout |

---

## Dataset Tolerances (Integration Expectations)

| Dataset | Description | Coordinate Tolerance | Distance Tolerance |
| --- | --- | --- | --- |
| PA3-A | Debug (low noise) | ±0.01 mm | ±0.01 mm |
| PA3-B | Debug (low noise) | ±0.01 mm | ±0.01 mm |
| PA3-C | Debug (moderate noise) | ±0.01 mm | ±0.01 mm |
| PA3-D | Debug (higher noise) | ±0.02 mm | ±0.02 mm |
| PA3-E | Debug (highest noise) | ±0.02 mm | ±0.02 mm |
| PA3-F | Debug (highest noise) | ±0.02 mm | ±0.02 mm |
| PA3-G/H/J | Unknown (smoke tests) | Relative assertions only (finite values, serialization round-trip) |

---

## High-Noise Dataset Comparisons (Outputs vs Expected)

### Worst-Case Rows

| Dataset | Row | Actual Sample (x, y, z) | Expected Sample | Δ Sample | Actual Closest (x, y, z) | Expected Closest | Δ Closest | Actual Dist | Expected Dist | Δ Dist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PA3-E | #9 | (‑10.90, 7.74, 52.02) | (‑10.89, 7.75, 52.03) | (‑0.01, ‑0.01, ‑0.01) | (‑7.00, 6.81, 51.90) | (‑6.99, 6.82, 51.90) | (‑0.01, ‑0.01, 0.00) | 4.02 | 4.01 | +0.01 |
| PA3-F | #10 | (4.20, ‑6.50, 63.39) | (4.21, ‑6.50, 63.39) | (‑0.01, 0.00, 0.00) | (4.14, ‑6.39, 63.12) | (4.15, ‑6.40, 63.11) | (‑0.01, 0.01, 0.01) | 0.30 | 0.30 | 0.00 |

### Snapshot: First Five Rows Per Dataset

**PA3-E**

| Row | Actual (Sample → Closest → Dist) | Expected | Δ |
| --- | --- | --- | --- |
| 1 | (17.61, ‑1.59, ‑27.45 → 20.80, ‑0.38, ‑24.98 → 4.21) | (17.61, ‑1.58, ‑27.45 → 20.80, ‑0.38, ‑24.98 → 4.21) | (0.00, ‑0.01, 0.00 → 0.00, 0.00, 0.00 → 0.00) |
| 2 | (‑30.72, 7.74, ‑31.33 → ‑31.29, 8.65, ‑31.76 → 1.16) | Same | (0.00 → all) |
| 3 | (‑32.04, ‑12.46, ‑49.83 → ‑31.31, ‑12.54, ‑46.86 → 3.06) | Same | (0.00 → all) |
| 4 | (‑37.54, ‑23.54, ‑42.78 → ‑34.96, ‑21.63, ‑40.19 → 4.13) | (‑37.54, ‑23.54, ‑42.78 → ‑34.96, ‑21.63, ‑40.19 → 4.12) | Δ Dist = +0.01 |
| 5 | (‑41.05, ‑19.62, ‑15.83 → ‑39.57, ‑19.02, ‑16.49 → 1.73) | (‑41.05, ‑19.62, ‑15.83 → ‑39.57, ‑19.01, ‑16.49 → 1.74) | Δ Closest y = ‑0.01, Δ Dist = ‑0.01 |

**PA3-F**

| Row | Actual (Sample → Closest → Dist) | Expected | Δ |
| --- | --- | --- | --- |
| 1 | (16.68, 24.06, 8.82 → 17.24, 25.73, 9.26 → 1.82) | (16.69, 24.07, 8.82 → 17.24, 25.73, 9.26 → 1.81) | Sample Δ = (‑0.01, ‑0.01, 0.00), Dist Δ = +0.01 |
| 2 | (30.10, ‑14.27, ‑12.22 → 28.12, ‑12.00, ‑12.86 → 3.08) | (30.10, ‑14.26, ‑12.22 → 28.13, ‑12.00, ‑12.86 → 3.07) | Sample/Closest shifts ≤0.01, Dist Δ = +0.01 |
| 3 | (‑6.96, 18.46, 40.21 → ‑6.93, 18.44, 40.18 → 0.04) | (‑6.95, 18.46, 40.20 → ‑6.93, 18.44, 40.18 → 0.04) | Sample Δ = (‑0.01, 0.00, +0.01) |
| 4 | (4.61, 0.16, ‑21.06 → 5.04, 0.56, ‑22.01 → 1.11) | (4.61, 0.17, ‑21.06 → 5.04, 0.56, ‑22.01 → 1.11) | Sample y Δ = ‑0.01 |
| 5 | (4.53, ‑19.81, ‑14.37 → 3.43, ‑17.87, ‑14.39 → 2.23) | Same | All zeros |

> Even in high-noise scenarios, the largest coordinate delta is at most 0.02 mm and the largest distance delta is 0.013 mm, staying within the configured tolerances.

---

## Execution & Reproducibility Checklist

| Task | Command |
| --- | --- |
| Full pipeline run | `python3 -m src.pa3 --mesh <mesh> --body-a <bodyA> --body-b <bodyB> --samples <samples> --output <out>` |
| CLI help | `python3 -m src.pa3 --help` |
| Full test suite | `python3 -m pytest tests -v` |
| Recompute and compare outputs | `python3 -m pytest tests/test_all_output_files.py -v` |
| View dataset diffs interactively | See scripts/commands used above |

---

## Notes

- The module structure relies on package-relative imports; use `from src.pa3 import …` or `python3 -m src.pa3` when invoking.
- Coverage tooling is optional; current evidence via regression diffs and tests indicates full feature parity with the original script.
- No outstanding TODOs, lint issues, or failing scenarios were observed.

---