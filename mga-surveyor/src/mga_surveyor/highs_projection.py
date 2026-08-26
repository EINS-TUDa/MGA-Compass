"""HiGHS (via raw highspy, no AML abstraction layer) reimplementation of
gurobi_projection.closest_point_on_hull -- same name, same contract
(point, points, norm) -> (c, normal), just this module's backend instead of
that one's; the two are never imported side by side under the same bare
name (pick one backend per solve), so there's no reason for one of them to
carry a `_highs` suffix the other doesn't. Mirrors the raw-gurobipy design
as closely as HiGHS's native
Python API allows: one persistent Highs() instance per (thread, norm),
extended with a new `w` column per new inner point via addCol, and the
query point folded into the bounds of two per-dimension rows (one per side
of the abs-value linearization) updated in place via changeRowBounds -- no
file export, no AML abstraction layer, matching the persistent-model
pattern that makes the Gurobi version fast (see project notes: an earlier
attempt through Pyomo's AML layer over the same persistent Gurobi backend
cost ~15-50x versus this, and linopy -- which has no persistent-solve path
at all -- cost 3,000-7,000x).

L1 and Linf only: HiGHS is an LP/MIP solver, no quadratic objective
support (confirmed separately: Pyomo's appsi_highs interface raises
DegreeError on a quadratic objective) -- no L2 branch here. Use
gurobi_projection.closest_point_on_hull for L2.

Tolerances are tightened to 1e-9 (HiGHS's default is 1e-7 for both primal
and dual feasibility), matching the Gurobi version's OptimalityTol/
FeasibilityTol=1e-9 -- important for iteration-by-iteration agreement with
the Gurobi backend over long runs, where looser tolerances would let small
per-call differences accumulate.

`highspy` is imported lazily, on first call to `closest_point_on_hull`,
not at module import time -- so `import mga_surveyor` (and picking
`backend='gurobi'` throughout) never requires highspy to be installed.
"""
import threading
from typing import Literal

import numpy as np

try:
    import highspy
except ImportError:
    highspy = None

VALID_NORMS_HIGHS = ("l1", "linf")
NormHighs = Literal["l1", "linf"]

_thread_local = threading.local()


def _require_highspy():
    if highspy is None:
        raise ImportError(
            "backend='highs' requires the `highspy` package, which is not "
            "installed here (pip install highspy). Use backend='gurobi' "
            "instead if you have a Gurobi license (see gurobi_projection."
            "closest_point_on_hull), or install highspy for a free, "
            "open-source alternative -- norm in ('l1', 'linf') only."
        )


def _build(norm: NormHighs, k: int, first_points: np.ndarray) -> dict:
    m0, k0 = first_points.shape
    assert k0 == k

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("primal_feasibility_tolerance", 1e-9)
    model.setOptionValue("dual_feasibility_tolerance", 1e-9)

    for i in range(m0):
        model.addCol(0.0, 0.0, highspy.kHighsInf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
    n_w = m0

    if norm == "linf":
        model.addCol(1.0, 0.0, highspy.kHighsInf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        aux_cols = [n_w]
    else:  # l1
        for j in range(k):
            model.addCol(1.0, 0.0, highspy.kHighsInf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        aux_cols = list(range(n_w, n_w + k))

    idx = np.arange(n_w, dtype=np.int32)
    val = np.ones(n_w, dtype=np.float64)
    model.addRow(1.0, 1.0, n_w, idx, val)
    sum_row = 0

    upper_rows, lower_rows = [], []
    row_idx = 1
    for j in range(k):
        aux = aux_cols[0] if norm == "linf" else aux_cols[j]
        cols = np.concatenate([np.arange(n_w, dtype=np.int32), [aux]]).astype(np.int32)
        vals_u = np.concatenate([first_points[:, j].astype(np.float64), [-1.0]])
        model.addRow(-highspy.kHighsInf, 0.0, len(cols), cols, vals_u)
        upper_rows.append(row_idx); row_idx += 1
        vals_l = np.concatenate([first_points[:, j].astype(np.float64), [1.0]])
        model.addRow(0.0, highspy.kHighsInf, len(cols), cols, vals_l)
        lower_rows.append(row_idx); row_idx += 1

    return {
        "model": model, "w_cols": list(range(n_w)), "aux_cols": aux_cols,
        "upper_rows": upper_rows, "lower_rows": lower_rows, "sum_row": sum_row,
        "m_known": m0, "n_cols": n_w + len(aux_cols),
    }


def closest_point_on_hull(point: np.ndarray, points: np.ndarray, norm: NormHighs) -> tuple[np.ndarray, np.ndarray]:
    _require_highspy()
    if norm not in VALID_NORMS_HIGHS:
        raise NotImplementedError(
            f"HiGHS backend is LP-only (no quadratic objective) -- norm must be one of "
            f"{VALID_NORMS_HIGHS}, got {norm!r}. Use gurobi_projection.closest_point_on_hull for L2."
        )
    point = np.asarray(point, dtype=float)
    points = np.asarray(points, dtype=float)
    m_target, k = points.shape

    attr = f"highs_state_{norm}"
    state = getattr(_thread_local, attr, None)
    if state is None:
        state = _build(norm, k, points[:m_target])
        setattr(_thread_local, attr, state)

    model = state["model"]
    w_cols, upper_rows, lower_rows = state["w_cols"], state["upper_rows"], state["lower_rows"]
    m_known = state["m_known"]

    if m_target > m_known:
        for p_new in points[m_known:m_target]:
            row_idx = np.array([state["sum_row"]] + upper_rows + lower_rows, dtype=np.int32)
            row_val = np.concatenate([[1.0], p_new.astype(np.float64), p_new.astype(np.float64)]).astype(np.float64)
            model.addCol(0.0, 0.0, highspy.kHighsInf, len(row_idx), row_idx, row_val)
            w_cols.append(state["n_cols"])
            state["n_cols"] += 1
        state["m_known"] = m_target

    for j in range(k):
        model.changeRowBounds(upper_rows[j], -highspy.kHighsInf, float(point[j]))
        model.changeRowBounds(lower_rows[j], float(point[j]), highspy.kHighsInf)

    status = model.run()
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS solve failed: status {status}")

    sol = model.getSolution()
    all_col_values = np.array(sol.col_value)
    w_val = all_col_values[w_cols]
    c_point = w_val @ points[:m_target]

    duals = model.allConstrDuals()
    normal = np.array([duals[upper_rows[j]] + duals[lower_rows[j]] for j in range(k)])
    return c_point, normal
