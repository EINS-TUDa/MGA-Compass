"""Gurobi (via gurobipy) implementation of the closest-point-on-hull
projection query -- `(point, points, norm) -> (c, normal)` -- the one
solver-backed operation `common.py` deliberately does not contain. Peer
module to `highs_projection.py`: same contract, different backend. Use this
one for all three norms, including L2 (a quadratic objective, which HiGHS
cannot solve); use `highs_projection.closest_point_on_hull` for an
open-source, license-free alternative when your norm is 'l1' or 'linf'.

`gurobipy` is imported lazily, on first call to `closest_point_on_hull`,
not at module import time -- so `import mga_surveyor` (and picking
`backend='highs'` throughout) never requires gurobipy to be installed, nor
pays for opening a licensed Gurobi environment it's never going to use.
"""
import threading

import numpy as np

from .common import Norm, check_norm

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

_thread_local = threading.local()
_gurobi_env = None
_gurobi_env_lock = threading.Lock()


def _require_gurobipy():
    if gp is None:
        raise ImportError(
            "backend='gurobi' requires the `gurobipy` package and a valid "
            "Gurobi license, neither of which is available here (pip install "
            "gurobipy). Use backend='highs' instead for a free, open-source "
            "alternative -- supports norm in ('l1', 'linf') only, see "
            "highs_projection.closest_point_on_hull."
        )


def _get_gurobi_env():
    global _gurobi_env
    if _gurobi_env is None:
        with _gurobi_env_lock:
            if _gurobi_env is None:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                _gurobi_env = env
    return _gurobi_env


def closest_point_on_hull(point: np.ndarray, points: np.ndarray, norm: Norm) -> tuple[np.ndarray, np.ndarray]:
    """Project `point` onto conv(points) under `norm` ('l1', 'l2', or
    'linf'), formulated as a convex combination of `points` -- no
    ConvexHull/facet enumeration needed, so this scales to the point counts
    that accumulate over an exploration run.

    Solved with a persistent, per-(thread, norm) Gurobi model built once
    and only ever updated: the query point enters only via the RHS of k
    residual constraints (O(k) per query), and new inner points are folded
    in with O(k) work each via `gp.Column` rather than a full rebuild.
    `Threads=1` since many of these are typically solved concurrently from
    a thread pool -- letting each individual model also spawn worker
    threads would oversubscribe the machine.

    Returns `(c, normal)`:
      - `c`: the closest point itself.
      - `normal`: a valid separating-hyperplane normal at `c`, i.e. every
        point `x` in the hull satisfies `normal @ (x - c) <= 0`, taken from
        the (sign-corrected) dual values of the residual constraints. For
        L2 this coincides with the residual `point - c` (L2 is self-dual);
        for L1/Linf the residual is *not* generally a valid normal, so the
        dual is required in general -- using the dual uniformly is what
        makes this function correct for all three norms with one code path.
    """
    _require_gurobipy()
    check_norm(norm)
    point = np.asarray(point, dtype=float)
    points = np.asarray(points, dtype=float)
    m_target, k = points.shape

    attr = f"qp_state_{norm}"
    state = getattr(_thread_local, attr, None)
    if state is None:
        model = gp.Model(env=_get_gurobi_env())
        model.Params.OutputFlag = 0
        model.Params.OptimalityTol = 1e-9
        model.Params.FeasibilityTol = 1e-9
        model.Params.NumericFocus = 3
        model.Params.Threads = 1

        sum_constr = model.addConstr(gp.LinExpr() == 1.0, name="sum_to_one")
        r_vars = [model.addVar(lb=-GRB.INFINITY, name=f"r{j}") for j in range(k)]
        resid_constrs = [model.addConstr(r_vars[j] == 0.0, name=f"resid{j}") for j in range(k)]

        if norm == "l2":
            model.Params.BarConvTol = 1e-12
            model.setObjective(gp.quicksum(r_vars[j] * r_vars[j] for j in range(k)), GRB.MINIMIZE)
        elif norm == "l1":
            s_vars = [model.addVar(lb=0.0, name=f"s{j}") for j in range(k)]
            for j in range(k):
                model.addConstr(s_vars[j] >= r_vars[j])
                model.addConstr(s_vars[j] >= -r_vars[j])
            model.setObjective(gp.quicksum(s_vars), GRB.MINIMIZE)
        else:  # linf
            t_var = model.addVar(lb=0.0, name="t")
            for j in range(k):
                model.addConstr(t_var >= r_vars[j])
                model.addConstr(t_var >= -r_vars[j])
            model.setObjective(t_var, GRB.MINIMIZE)

        model.update()
        state = {
            "model": model, "w_vars": [], "sum_constr": sum_constr,
            "resid_constrs": resid_constrs, "m_known": 0,
        }
        setattr(_thread_local, attr, state)

    model = state["model"]
    w_vars = state["w_vars"]
    sum_constr = state["sum_constr"]
    resid_constrs = state["resid_constrs"]
    m_known = state["m_known"]

    if m_target > m_known:
        new_points = points[m_known:m_target]
        for offset, p_new in enumerate(new_points):
            col = gp.Column([1.0], [sum_constr])
            var = model.addVar(lb=0.0, column=col, name=f"w{m_known + offset}")
            model.update()
            for j in range(k):
                model.chgCoeff(resid_constrs[j], var, -float(p_new[j]))
            w_vars.append(var)
        state["m_known"] = m_target
        model.update()

    # bulk RHS update -- one C API call for all k constraints, instead of
    # k individual .RHS = ... assignments (see project notes: this and the
    # bulk result reads below were added specifically to isolate whether an
    # observed HiGHS speed edge was a genuine solver-algorithm difference or
    # just an artifact of per-element vs bulk Python attribute access).
    model.setAttr("RHS", resid_constrs, [-float(point[j]) for j in range(k)])

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi solve failed: status {model.Status}")

    # bulk result reads -- one C API call each for all m variables / all k
    # constraints, instead of m + k individual .X / .Pi attribute accesses.
    w = np.array(model.getAttr("X", w_vars))
    c = w @ points[:m_target]
    normal = -np.array(model.getAttr("Pi", resid_constrs))
    return c, normal
