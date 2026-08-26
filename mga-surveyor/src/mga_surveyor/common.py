"""Norm-generic geometric primitives usable by any worst-point method -- not
tied to one method's specific algorithm, unlike e.g. `halfspaces_to_scipy`/
`compute_active_constrs`, which live in `methods/vertex_enum.py` instead
because only `VertexEnumMethod`'s incremental vertex-maintenance actually
needs them.

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
