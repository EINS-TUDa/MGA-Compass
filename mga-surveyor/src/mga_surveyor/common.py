"""Norm-generic geometric primitives usable by any worst-point method -- not
tied to one method's specific algorithm.

Everything here operates only on point clouds and halfspace lists handed to
it -- nothing in this module knows anything about where those points or
cuts came from (a real system model, a synthetic test polytope, or
anything else), and nothing here is tied to a particular solver backend
either. A halfspace/cut is always represented as a `(direction, point)`
pair meaning `{x : direction @ (x - point) <= 0}`; this is the one
convention used consistently across the whole package.

The one operation this package needs that genuinely requires a solver --
projecting a point onto conv(points) under a given norm -- deliberately
does NOT live here. It lives in `gurobi_projection.py` and `highs_projection.py`
instead, one module per backend, both implementing the same
`(point, points, norm) -> (c, normal)` contract, so this module can stay
backend-agnostic in fact, not just in name.
"""
from typing import Literal

import numpy as np

VALID_NORMS = ("l1", "l2", "linf")

#: One of the three supported distance metrics.
Norm = Literal["l1", "l2", "linf"]
#: A `(direction, point)` pair meaning `{x : direction @ (x - point) <= 0}`.
Halfspace = tuple[np.ndarray, np.ndarray]


def check_norm(norm: str) -> None:
    """Validate an untrusted `norm` value at a package boundary.

    Deliberately typed as plain ``str``, not :data:`Norm` -- the whole
    point of this function is to check a value that isn't yet known to be
    one of the three valid options; every other function in the package
    that already expects a valid norm uses :data:`Norm` instead.
    """
    if norm not in VALID_NORMS:
        raise ValueError(f"norm must be one of {VALID_NORMS}, got {norm!r}")


def gaps(outer_vertices: np.ndarray, closest_points: np.ndarray, norm: Norm) -> np.ndarray:
    """Distance from each outer vertex to its closest point on the inner
    hull, under the SAME norm used for the projection -- so the argmax
    selection and the stopping tolerance measure the thing that was
    actually optimized."""
    check_norm(norm)
    diff = np.asarray(outer_vertices) - np.asarray(closest_points)
    if norm == "l1":
        return np.abs(diff).sum(axis=1)
    if norm == "linf":
        return np.abs(diff).max(axis=1)
    return np.linalg.norm(diff, axis=1)


def linearized_objective_cut(
    x0: np.ndarray, obj_value: float, duals: np.ndarray, obj_optimal: float, slack: float, atol: float = 1e-9,
) -> Halfspace | None:
    """A supporting halfspace of the true near-optimal region, linearized
    from a single solved point's own duals -- valid, not a heuristic: the
    minimum-cost-achievable-at-x value function is convex in x (a standard
    parametric-LP fact), so its tangent at any solved point x0 is a GLOBAL
    underestimator, `obj(x) >= obj_value + duals @ (x - x0)` for every x,
    not just near x0. Near-optimality (`obj(x) <= (1 + slack) *
    obj_optimal`) therefore forces `duals @ (x - x0) <= K` where `K =
    (1 + slack) * obj_optimal - obj_value` (>= 0, since x0 -- the point
    the duals came from -- is itself near-optimal).

    Returned as this package's usual `(direction, point)` halfspace
    convention: `direction = duals`, and `point` is `x0` shifted along a
    single axis -- whichever one has `duals`' own largest-magnitude
    component, `j = argmax(|duals|)` -- by just enough to make `duals @
    (point - x0) == K` exactly: `point = x0 + (K / duals[j]) * e_j`.
    Collapses to `x0` itself when `K == 0` (x0 has no slack left), same
    as any other cut that passes through its own generating point.

    Deliberately NOT the Euclidean projection (`x0 + (K / (duals @
    duals)) * duals`) despite that also being a valid choice of point on
    the same halfspace (`add_cut`'s own halfspace only depends on the
    scalar `direction @ point`, identical for every point actually on the
    plane -- so this is still just a choice among equally-correct
    options, not a different halfspace). `duals` mixes real-world units
    that can differ by orders of magnitude (e.g. $/MW vs $/MWh), so
    `duals @ duals` can be dominated by whichever dimension happens to
    have large values there, sending the Euclidean-projection point to a
    numerically extreme value in every OTHER dimension -- which then
    makes `direction @ point` (computed downstream by whichever method
    processes this halfspace) suffer catastrophic floating-point
    cancellation, corrupting that cut's own vertex classification
    (confirmed in practice, against this project's incremental edge-walking
    construction before `CddMethod` became the production method: exactly
    this construction once produced ~187,000 spurious crossing vertices for
    a 4-dimensional polytope, an OOM crash, not just an inefficiency).
    Dividing by `duals[j]` itself
    (the largest-magnitude component, so the least numerically sensitive
    denominator available) and moving along a single axis avoids ever
    amplifying small numerical noise into a huge coordinate swing.

    Returns `None` when `duals` is (numerically) the zero vector: then
    `K >= 0` is the whole constraint, true for every x -- a vacuous cut,
    not worth adding.
    """
    x0 = np.asarray(x0, dtype=float)
    duals = np.asarray(duals, dtype=float)
    j = int(np.argmax(np.abs(duals)))
    if abs(duals[j]) < atol:
        return None
    k = max(0.0, (1 + slack) * obj_optimal - obj_value)
    point = x0.copy()
    point[j] += k / duals[j]
    return duals, point
