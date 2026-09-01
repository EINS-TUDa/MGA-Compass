"""Reference implementations of the two "true region" queries that
`mga_surveyor`'s explore.drive needs from something outside the package --
against an EXPLICIT H-represented polytope (`A_ub x <= b_ub`), via LP/QP.

These live in `mga_surveyor/tests/`, not in the core package modules
(`common.py`, `methods/`, `explore.py`), because having an explicit
`(A_ub, b_ub)` at all is itself a test-example detail. A real system model
(e.g. a PyPSA network) never hands you that -- it answers the same two
queries by solving its own optimization problem (a linopy/Gurobi/HiGHS
model over its actual constraints) instead of an LP against a polytope
someone already enumerated. None of the core package modules import this
one; everything in `mga_surveyor/common.py` operates only on point clouds
and halfspace lists already handed to it, independent of where they came
from. Swap these two functions for calls into a real model to run the same
methods/explore.drive on a real problem.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from mga_surveyor.common import check_norm

_gurobi_env = gp.Env(empty=True)
_gurobi_env.setParam("OutputFlag", 0)
_gurobi_env.start()


def furthest_point(direction, A_ub, b_ub, A_eq=None, b_eq=None, bounds=None):
    """Support-function LP query: the point in {x : A_ub x <= b_ub,
    A_eq x == b_eq} maximizing direction @ x. Used only for bootstrapping
    -- seeding the initial O / I along the axis directions (see
    `synthetic_ball.bootstrap_state`). A direction-based variant of the
    true-region query that used to call this too has been removed (see
    `mga_surveyor.explore.drive`'s docstring); that query now always uses
    `closest_point_in_region` below instead.
    """
    from scipy.optimize import linprog

    direction = np.asarray(direction, dtype=float)
    if bounds is None:
        bounds = [(None, None)] * direction.shape[0]
    res = linprog(c=-direction, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x


def closest_point_in_region(target, A_ub, b_ub, norm):
    """The point in the H-represented region `{x : A_ub x <= b_ub}`
    closest to `target` under `norm`, together with a valid
    separating-hyperplane normal at that point -- generalized from Linf
    (the only norm this construction was originally worked out for) to any
    of the three norms via the same dual-of-the-residual trick used in
    `mga_surveyor.gurobi_projection.closest_point_on_hull` (see that
    function's docstring for why the dual, not the raw residual, is needed
    for L1/Linf).

    Returns `(point, normal)`. If `target` is itself feasible, `point ==
    target` (distance zero) and no cut should be added -- callers should
    check `||target - point|| ~ 0` before deciding to add a cut with
    `normal`.
    """
    check_norm(norm)
    target = np.asarray(target, dtype=float)
    k = len(target)

    model = gp.Model(env=_gurobi_env)
    model.Params.OutputFlag = 0
    model.Params.OptimalityTol = 1e-9
    model.Params.FeasibilityTol = 1e-9

    x_vars = [model.addVar(lb=-GRB.INFINITY, name=f"x{j}") for j in range(k)]
    r_vars = [model.addVar(lb=-GRB.INFINITY, name=f"r{j}") for j in range(k)]
    # residual defined as target - x, not x - target -- the sign of the
    # dual-derived normal below depends on this choice.
    resid_constrs = [model.addConstr(r_vars[j] == target[j] - x_vars[j], name=f"resid{j}") for j in range(k)]

    for row in range(A_ub.shape[0]):
        model.addConstr(gp.quicksum(A_ub[row, j] * x_vars[j] for j in range(k)) <= b_ub[row])

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

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi solve failed: status {model.Status}")

    point = np.array([x_vars[j].X for j in range(k)])
    normal = np.array([rc.Pi for rc in resid_constrs])
    return point, normal
