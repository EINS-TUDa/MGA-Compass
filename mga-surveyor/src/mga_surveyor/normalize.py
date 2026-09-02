"""Generic affine-map normalization: keeps a `Method`'s inner/outer
polytope confined to the unit cube `[-1, 1]^k`, regardless of how
differently-scaled the underlying dimensions are (e.g. MW vs MWh), by
wrapping `explore.drive`'s three oracle callbacks
(`FurthestPointFn`/`ClosestPointFn`/`OptimalPointFn`) with a per-dimension
affine map computed from the box points -- `explore.drive`/`Method` need
no changes at all; they just see oracle functions that happen to speak
normalized coordinates instead of real ones.

Finding the actual closest point under this normalized geometry isn't
just a coordinate relabeling, though: a plain L1/Linf objective over raw
real-unit residuals doesn't correspond to the cube-normalized notion of
distance unless each dimension's residual is weighted by the same
per-dimension scale this module computes (`AffineMap.weights`) --
`normalize_oracle_fns` only computes that scale, the actual weighting has
to happen inside whatever solves the closest-point problem itself (see
`mga_surveyor.pypsa_ext.make_closest_point_fn`'s own `weights` parameter).
Without it, the point found would be closest under real-unit distance,
not the cube-normalized distance `Method` is actually exploring under.

None of this is PyPSA-specific -- same independence as `explore.py`
itself, so it's usable against `mga_surveyor.tests.lp_region` or any
other oracle implementation too.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .explore import ClosestPointFn, FurthestPointFn, OptimalPointFn, box_directions


@dataclass(frozen=True)
class AffineMap:
    """Per-dimension affine map real <-> normalized:
    `normalized = weights * (real - center)`, where `weights = 1 /
    half_range` -- calibrated so `[center - half_range, center +
    half_range]` maps exactly onto `[-1, 1]` per dimension.
    """
    center: np.ndarray
    half_range: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        """`1 / half_range` -- what a point's own coordinates get scaled
        by (`to_normalized`/`to_real`), and what a closest-point solve's
        objective needs to weight each dimension's residual by for
        "closest" to mean the same thing in real units as it does in this
        normalized cube (see module docstring)."""
        return 1.0 / self.half_range

    def to_normalized(self, point: np.ndarray) -> np.ndarray:
        return self.weights * (np.asarray(point, dtype=float) - self.center)

    def to_real(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=float) / self.weights + self.center

    def normal_to_normalized(self, normal: np.ndarray) -> np.ndarray:
        """A halfspace normal transforms by the *inverse* of what a point
        does (`half_range`, not `weights`) -- it's a dual vector (a
        gradient), not a point, so it transforms by the map's
        inverse-transpose, which for this diagonal map is just its
        inverse again. Substituting `x = to_real(x')` into a real-space
        halfspace `{x : normal . (x - p) <= 0}` gives exactly `{x' :
        normal_to_normalized(normal) . (x' - to_normalized(p)) <= 0}` --
        the same halfspace, re-expressed in normalized coordinates."""
        return np.asarray(normal, dtype=float) * self.half_range


def compute_affine_map(box_points: np.ndarray, keys: list[str] | None = None, atol: float = 1e-9) -> AffineMap:
    """From the `2k` box-direction points `explore.drive`'s bootstrap
    already computes (rows `0..k-1` = the `+e_i` maximizers, rows
    `k..2k-1` = the `-e_i` minimizers -- `explore.drive`'s own
    `box_directions` ordering), read off each dimension's true achievable
    `[min, max]` directly: row `i`'s own `i`-th coordinate IS that
    dimension's max (maximizing `e_i . x` maximizes `x_i` alone, nothing
    else), row `k+i`'s IS its min. Calibrated exactly, not approximately
    -- each dimension's achievable range maps onto exactly `[-1, 1]`.

    `keys` (default `None`): dimension names, used only to name the
    offending dimension(s) in the error below -- purely cosmetic,
    optional so this stays usable without them.

    :raises ValueError: If any dimension's achievable range is (numerically)
        zero -- fixed at a single value under the current slack budget.
        There's nothing meaningful to rescale by for a fixed dimension
        (`weights = 1 / half_range` would be a division by zero), and a
        dimension with no achievable range at all has no reason to be in
        `dimensions` in the first place -- likely the underlying network
        component isn't actually extendable. Raised (not silently worked
        around) so this fails loudly instead of quietly producing
        `inf`/`nan` downstream.
    """
    box_points = np.asarray(box_points, dtype=float)
    k = box_points.shape[0] // 2
    maxes = np.array([box_points[i, i] for i in range(k)])
    mins = np.array([box_points[k + i, i] for i in range(k)])
    half_range = (maxes - mins) / 2

    degenerate = np.where(half_range <= atol)[0]
    if len(degenerate):
        names = [keys[i] if keys is not None else f"dimension {i}" for i in degenerate]
        msg = (
            f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} fixed at a single value "
            f"under the current slack budget (no achievable range) -- normalize=True has nothing "
            f"meaningful to rescale by for a fixed dimension. Remove it from `dimensions` (e.g. make "
            f"the underlying component extendable) or run without normalize=True."
        )
        raise ValueError(msg)

    return AffineMap(center=(maxes + mins) / 2, half_range=half_range)


def presolve_box_points(furthest_point: FurthestPointFn, k: int) -> tuple[FurthestPointFn, np.ndarray, np.ndarray]:
    """Solve the `2k` box directions exactly once, right now, and return a
    `FurthestPointFn` that just returns this same already-computed result
    instead of re-solving -- for any caller that needs the box points (or
    needs whatever side effects that one solve has, e.g. an `on_solve`
    callback baked into `furthest_point` itself -- see
    `mga_surveyor.pypsa_ext.make_closest_point_fn`'s `use_objective_cuts`
    caller) before `explore.drive`'s own bootstrap gets to them, without
    `explore.drive` ending up re-solving (and re-triggering those side
    effects) a second time for nothing.
    """
    directions = box_directions(k)
    box_points, box_normals = furthest_point(directions)
    box_points = np.asarray(box_points, dtype=float)
    box_normals = np.asarray(box_normals, dtype=float)

    def furthest_point_reused(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Only ever called by drive() with exactly these same
        # box_directions (its docstring: "called once ... for
        # bootstrapping only") -- nothing to resolve.
        del directions
        return box_points, box_normals

    return furthest_point_reused, box_points, box_normals


def normalize_from_box_points(
    box_points: np.ndarray,
    box_normals: np.ndarray,
    make_closest_point: Callable[[np.ndarray], ClosestPointFn],
    optimal_fn: OptimalPointFn,
    keys: list[str] | None = None,
) -> tuple[FurthestPointFn, ClosestPointFn, OptimalPointFn, AffineMap]:
    """The rest of `normalize_oracle_fns`, split out for a caller that
    already has `box_points`/`box_normals` in hand (e.g. from its own
    `presolve_box_points` call, made for some other reason too -- like
    filtering candidate halfspaces against the box ones before they ever
    reach `Method.add_cut`) and doesn't want a second, redundant solve of
    the same `2k` directions.

    `make_closest_point(weights) -> ClosestPointFn` is a factory, not a
    ready-made `ClosestPointFn`, because the right weights (this
    `AffineMap`'s own `weights`) are only known once `box_points` are in
    hand -- see `mga_surveyor.pypsa_ext.make_closest_point_fn`'s own
    `weights` parameter for what those actually do to its objective.

    `keys` (default `None`): dimension names, passed straight through to
    `compute_affine_map` -- purely cosmetic, only used to name the
    offending dimension if it raises.

    Returns `(furthest_point_n, closest_point_n, optimal_fn_n,
    affine_map)` -- the wrapped trio, plus the `AffineMap` itself for a
    caller that needs to convert points already in real units (e.g.
    `extra_points`/`extra_halfspaces` from an earlier, interrupted run)
    into this same normalized space before folding them into
    `explore.drive`'s bootstrap.

    :raises ValueError: If any dimension has no achievable range (see
        `compute_affine_map`).
    """
    affine_map = compute_affine_map(box_points, keys)

    closest_point = make_closest_point(affine_map.weights)

    def furthest_point_n(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del directions
        return (
            np.array([affine_map.to_normalized(p) for p in box_points]),
            np.array([affine_map.normal_to_normalized(n) for n in box_normals]),
        )

    def closest_point_n(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point, normal = closest_point(affine_map.to_real(target))
        return affine_map.to_normalized(point), affine_map.normal_to_normalized(normal)

    def optimal_fn_n() -> np.ndarray:
        return affine_map.to_normalized(optimal_fn())

    return furthest_point_n, closest_point_n, optimal_fn_n, affine_map


def normalize_oracle_fns(
    furthest_point: FurthestPointFn,
    make_closest_point: Callable[[np.ndarray], ClosestPointFn],
    optimal_fn: OptimalPointFn,
    k: int,
    keys: list[str] | None = None,
) -> tuple[FurthestPointFn, ClosestPointFn, OptimalPointFn, AffineMap]:
    """Wrap the three `explore.drive` oracle callbacks so they speak
    normalized `[-1, 1]^k` coordinates instead of real units --
    `explore.drive`/`Method` need no changes at all, they just see
    functions matching the same three protocols.

    Solves the `2k` box directions exactly once, here (`presolve_box_points`)
    -- `explore.drive`'s own bootstrap never needs to. A thin convenience
    wrapper over `presolve_box_points` + `normalize_from_box_points`, for
    a caller that doesn't need `box_points`/`box_normals` for anything
    else; see `normalize_from_box_points` for one that does.

    :raises ValueError: If any dimension has no achievable range (see
        `compute_affine_map`).
    """
    _furthest_point_reused, box_points, box_normals = presolve_box_points(furthest_point, k)
    return normalize_from_box_points(box_points, box_normals, make_closest_point, optimal_fn, keys)
