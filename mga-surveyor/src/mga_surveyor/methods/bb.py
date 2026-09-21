"""``BBMethod``.

- Uses SciPy's ``ConvexHull`` (Qhull) on the polar dual of the outer
  polytope's halfspaces to compute the outer polytope's vertex set
  (H-to-V enumeration) -- an alternative engine to :class:`~mga_surveyor.
  methods.cdd.CddMethod`'s cddlib Double Description method. See that
  module's docstring for the general ``Method`` contract this mirrors;
  this class implements the same ``bootstrap``/``add_point``/``add_cut``/
  ``find_trial_point``/``max_gap`` interface, just backed by a different
  vertex-enumeration algorithm.

- Named for the incremental algorithm, not the one-off initial build:
  Qhull only runs its usual divide-and-conquer quickhull ONCE, the first
  time enough dual points have accumulated to form an initial simplex
  hull (see :meth:`_add_halfspace`). Every subsequent dual point --
  i.e. every :meth:`add_cut` for the rest of this object's life, which in
  production usage is the overwhelming majority of calls -- goes through
  ``ConvexHull.add_points`` on that already-built hull, which Qhull
  documents as switching to a beneath-and-beyond insertion (find the
  facets the new point sees "beyond", remove them, stitch in new facets
  from the point to the resulting horizon ridge -- see Preparata & Shamos)
  rather than rerunning quickhull from scratch. That incremental step is
  what this class actually spends its time doing, hence the name.

- The polar-dual construction requires a fixed reference point ``x0``
  strictly inside every halfspace registered so far (a standard
  requirement of this technique -- see e.g. Preparata & Shamos): a
  halfspace ``{x : direction@(x-point) <= 0}``, recentered at ``x0``, maps
  to the dual point ``direction / b`` where ``b = direction@(point-x0)``,
  which is only well-defined (and the resulting dual facet/primal vertex
  correspondence only valid) when ``b > 0``, i.e. ``x0`` strictly satisfies
  the halfspace. By default, :meth:`BBMethod.bootstrap` uses the
  centroid (mean) of ``initial_points`` as ``x0`` and keeps it fixed for
  the method's lifetime; pass ``origin`` to :meth:`__init__` instead when
  a specific point is *known* to be strictly (and durably, not just
  barely) interior -- see that parameter's own docstring. Either way this
  is a heuristic, not a guarantee: the ``Method`` interface doesn't
  designate any one of ``initial_points`` as "the" interior reference
  point, and no fixed choice is guaranteed interior for every possible
  caller. If a later :meth:`add_cut` (or the initial halfspaces
  themselves) violates ``x0``'s interiority, :func:`_dual_point` raises
  ``ValueError`` immediately rather than silently returning a wrong vertex
  set -- in practice this hasn't triggered for the box-plus-optimal-point
  bootstrap pattern :mod:`mga_surveyor.explore` actually uses, since the
  centroid of a box's extreme points plus one interior (near-optimal)
  point lands well inside the box.

- As with :func:`~mga_surveyor.methods.cdd._halfspace_row`, each
  halfspace's ``direction`` is unit-normalized (divided by its own
  Euclidean norm) before its dual point is computed. Scaling a halfspace's
  defining vector by any positive constant doesn't change the halfspace
  itself, so this is free geometrically; it matters here for the same
  reason it matters for cdd's float backend: Qhull's own degeneracy/
  coplanarity tests compare against a fixed relative tolerance internally,
  and cuts arriving at wildly different natural magnitudes (as they do on
  this project's real networks) make that tolerance inconsistently loose
  or tight from cut to cut. Putting every cut on the same (unit) scale
  before dualizing removes that inconsistency.

- Unlike ``CddMethod``, Qhull's incremental ``add_points`` doesn't expose
  any persistent per-vertex identity across an update (cddlib's patched
  ``copy_generators_with_ids`` has no Qhull equivalent), so this class
  can't tell "this is the same vertex as before the last cut" by a true
  id. It approximates one instead: :func:`_vertex_hash` rounds a vertex's
  coordinates to :data:`_VERTEX_HASH_DECIMALS` places and uses the
  resulting tuple as a stand-in identity, on the empirical grounds (see
  ``qhull_survival_check.py`` in project notes) that on this project's
  real networks, 80-98% of a refresh's vertices reproduce bit-identical
  coordinates (well past 1e-9, tighter than the 1e-8 used here) from one
  ``add_cut`` to the next -- only the vertices a cut actually creates or
  destroys change at all. :meth:`_refresh` uses this hash exactly the way
  ``CddMethod._refresh`` uses a real id: build a hash->index lookup from
  the PREVIOUS refresh's vertices, and for the new vertex set, reuse the
  cached closest point/normal for every hash that survived unchanged,
  solving ``closest_point_on_hull`` only for the ones that didn't. A
  false-positive hash collision (two geometrically distinct vertices
  rounding to the same tuple) would silently reuse the wrong cached
  value -- :data:`_VERTEX_HASH_DECIMALS` is chosen with headroom below
  the projection solver's own ``1e-9`` tolerances precisely to make that
  not happen in practice; a false negative (a survived vertex hashing
  differently by tiny roundoff) only costs an avoidable re-solve, never
  correctness. :meth:`add_point` still uses its own, separate,
  non-hash-based reuse for the same reason it always has -- growing the
  inner approximation never changes the outer vertex set at all, so
  :meth:`_refresh_stale_closest_points` can just re-use the existing
  arrays directly, no identity of any kind needed.
"""
from concurrent.futures import Executor
from typing import Literal, Protocol

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..common import Halfspace, Norm, gaps, check_norm
from ..gurobi_projection import closest_point_on_hull
from .base import Method


class ProjectionFn(Protocol):
    """Contract for `projection_fn` -- see `mga_surveyor.methods.cdd.ProjectionFn`."""
    def __call__(self, point: np.ndarray, points: np.ndarray, norm: Norm) -> tuple[np.ndarray, np.ndarray]: ...


#: Rounding precision used by :func:`_vertex_hash` -- see the module
#: docstring's note on hash-based vertex identity. Matches
#: ``_compute_outer_vertices``'s own internal dedup rounding, so hashing
#: its output loses no additional precision beyond what that step already
#: commits to.
_VERTEX_HASH_DECIMALS = 8


def _vertex_hash(vertex: np.ndarray) -> tuple:
    """A vertex's coordinates, rounded to :data:`_VERTEX_HASH_DECIMALS`
    places, as a hashable stand-in for a persistent id -- see module
    docstring.
    """
    return tuple(np.round(vertex, decimals=_VERTEX_HASH_DECIMALS))


def _dual_point(direction: np.ndarray, point: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Maps the halfspace ``{x : direction@(x-point) <= 0}`` to its polar
    dual point relative to ``origin``.

    ``direction`` is unit-normalized first -- see the module docstring.
    Recentering at ``origin`` rewrites the halfspace as
    ``direction@(x-origin) <= direction@(point-origin)``, i.e. ``A@y <= b``
    with ``y = x-origin``, ``A = direction``, ``b = direction@(point-origin)``;
    the dual point is ``A/b``.

    :raises ValueError: if ``b <= 0``, i.e. ``origin`` doesn't lie strictly
        inside this halfspace -- meaning ``origin`` isn't a valid interior
        reference point for the dual construction (see module docstring).
    """
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm
    b = float(direction @ (np.asarray(point, dtype=float) - origin))
    if b <= 0:
        raise ValueError(
            f"origin must lie strictly inside every halfspace for the polar-dual "
            f"construction; got b={b} for this one (origin is on the boundary or "
            f"outside it)"
        )
    return direction / b


class BBMethod(Method):
    """See module docstring.

    :param k: Dimension of the exploratory variables.
    :param norm: Distance metric, one of ``'l1'``, ``'l2'``, or ``'linf'``.
    :type norm: str
    :param executor: Used to run the independent per-vertex
        ``closest_point_on_hull`` solves concurrently after each refresh.
        A fresh ``ThreadPoolExecutor`` is created if none is given.
    :type executor: concurrent.futures.Executor, optional
    :param projection_fn: Lower-level override of the inner-hull projection
        callable itself -- takes precedence over ``projection_backend`` if
        given. See ``CddMethod``'s own docstring for the exact contract.
    :param projection_backend: Which persistent solver to use for the
        inner-hull projection -- ``'gurobi'`` (default) or ``'highs'``.
    :type projection_backend: str
    :param origin: Fixed interior reference point for the polar-dual
        construction (see module docstring) -- overrides the default
        centroid-of-``initial_points`` heuristic :meth:`bootstrap` would
        otherwise use. Pick this explicitly whenever a specific point is
        *known* to be strictly interior to every halfspace for the whole
        run (e.g. the cost-optimal point in :mod:`mga_surveyor.explore`'s
        own usage) -- the centroid of a caller's initial points is only a
        heuristic and isn't guaranteed interior, and even when it is,
        staying strictly interior isn't the same as staying *well inside*:
        as cuts accumulate and the true region shrinks around a FIXED
        origin over many iterations, some halfspace's ``b = direction@
        (point-origin)`` can shrink towards zero even while staying
        positive, degrading that halfspace's dual point ``direction/b``
        (and hence the vertices it participates in) numerically -- a
        plausible explanation if this class's vertex set/gap ever starts
        drifting from ``CddMethod``'s on the same problem after many
        iterations. A point known to stay comfortably interior throughout
        (rather than one recomputed once from whatever happens to be in
        ``initial_points``) avoids that. No default -- ``None`` keeps the
        centroid heuristic.
    :raises ValueError: If ``projection_backend`` is not ``'gurobi'`` or
        ``'highs'``.
    """

    def __init__(
        self, k: int, norm: Norm, executor: Executor | None = None,
        projection_fn: ProjectionFn | None = None, projection_backend: Literal["gurobi", "highs"] = "gurobi",
        origin: np.ndarray | None = None,
    ) -> None:
        check_norm(norm)
        super().__init__(k, norm)
        self._origin_override = None if origin is None else np.asarray(origin, dtype=float)
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=8)
        self.executor = executor
        self.projection_backend = projection_backend
        if projection_fn is not None:
            self.projection_fn = projection_fn
        elif projection_backend == "highs":
            from ..highs_projection import closest_point_on_hull as closest_point_on_hull_highs
            self.projection_fn = closest_point_on_hull_highs
        elif projection_backend == "gurobi":
            self.projection_fn = closest_point_on_hull
        else:
            raise ValueError(f"projection_backend must be 'gurobi' or 'highs', got {projection_backend!r}")

        self.inner_vertices = None
        self.outer_halfspaces = []
        self._origin = None
        self._dual_points = []  # aligned with outer_halfspaces
        self._hull = None  # incremental scipy.spatial.ConvexHull over self._dual_points, or None until enough points
        self._outer_vertices = None
        self._vertex_hashes = None  # hash-based stand-in ids, aligned with self._outer_vertices -- see module docstring
        self._closest_points = None
        self._closest_normals = None
        self._gap = None
        self._outer_stale = True  # True whenever the outer vertex set itself may have changed since last refresh
        self._dirty = True

    def bootstrap(self, initial_points: list[np.ndarray], initial_halfspaces: list[Halfspace]) -> None:
        self.inner_vertices = np.array(initial_points, dtype=float)
        self._origin = self._origin_override if self._origin_override is not None else self.inner_vertices.mean(axis=0)
        self.outer_halfspaces = []
        self._dual_points = []
        self._hull = None
        self._outer_vertices = None
        self._vertex_hashes = None
        self._closest_points = None
        self._closest_normals = None
        self._outer_stale = True
        self._dirty = True
        for direction, point in initial_halfspaces:
            self._add_halfspace(direction, point)

    def add_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        is_new = np.linalg.norm(self.inner_vertices - point, axis=1).min() > 1e-6
        if is_new:
            self.inner_vertices = np.vstack([self.inner_vertices, point])
            if self._closest_normals is not None:
                self._refresh_stale_closest_points(point)
            self._dirty = True
        return is_new

    def _refresh_stale_closest_points(self, new_inner_vertex: np.ndarray) -> None:
        """Identical in spirit to ``CddMethod``'s own: after the inner
        approximation grows, recompute only the cached closest-point/normal
        pairs that ``new_inner_vertex`` actually invalidates, not every
        vertex. Valid without any persistent vertex identity because the
        outer vertex set itself (``self._outer_vertices``, aligned with
        ``self._closest_points``/``self._closest_normals``) doesn't change
        between :meth:`add_cut` calls -- only :meth:`add_cut` rebuilds it.

        :param new_inner_vertex: The newly added inner vertex.
        """
        if self._outer_vertices is None or len(self._outer_vertices) == 0:
            return
        rhs = new_inner_vertex[None, :] - self._closest_points
        still_valid = np.einsum("ij,ij->i", self._closest_normals, rhs) <= 1e-9
        stale_positions = np.where(~still_valid)[0]
        if len(stale_positions) == 0:
            return
        stale_points = list(self._outer_vertices[stale_positions])
        results = list(self.executor.map(
            lambda p: self.projection_fn(p, self.inner_vertices, norm=self.norm), stale_points
        ))
        for pos, (c, n) in zip(stale_positions.tolist(), results):
            self._closest_points[pos] = c
            self._closest_normals[pos] = n

    def _add_halfspace(self, direction: np.ndarray, point: np.ndarray) -> None:
        direction = np.asarray(direction, dtype=float)
        point = np.asarray(point, dtype=float)
        dual = _dual_point(direction, point, self._origin)
        self.outer_halfspaces.append((direction, point))
        self._dual_points.append(dual)
        num_points = len(self._dual_points)
        if num_points <= self.k:
            return  # Qhull needs at least k+1 points to form an initial d-simplex hull
        if self._hull is None:
            try:
                self._hull = ConvexHull(np.array(self._dual_points), incremental=True)
            except QhullError:
                pass  # initial dual points are coplanar/degenerate; try again once more have accumulated
        else:
            self._hull.add_points(dual.reshape(1, -1))

    def add_cut(self, direction: np.ndarray, point: np.ndarray) -> None:
        self._add_halfspace(direction, point)
        self._outer_stale = True
        self._dirty = True

    def _compute_outer_vertices(self) -> np.ndarray:
        """The outer polytope's current vertex set, recovered from the
        dual hull's facets -- cheap: reads ``self._hull.equations`` as it
        already stands, doesn't trigger any new Qhull computation (that
        happens in :meth:`_add_halfspace` instead, via ``add_points``).
        """
        if self._hull is None:
            return np.zeros((0, self.k))
        equations = self._hull.equations  # shape (num_facets, k+1): [normal | offset]
        normals = equations[:, :-1]
        offsets = equations[:, -1]
        valid_mask = ~np.isclose(offsets, 0)  # a facet through the dual origin has no primal image
        primal = -normals[valid_mask] / offsets[valid_mask, np.newaxis]
        unique = np.unique(np.round(primal, decimals=_VERTEX_HASH_DECIMALS), axis=0)
        return unique + self._origin

    def _refresh(self) -> None:
        """Recompute the cached outer vertex set (if stale) and per-vertex
        closest inner points/gap (if the outer vertex set changed, or an
        inner point invalidated some of the cache) -- if stale, solves
        ``closest_point_on_hull`` only for vertices whose hash (see module
        docstring) is new since the last refresh, reusing the cached
        closest point/normal for every hash that survived unchanged --
        the same shape of optimization ``CddMethod._refresh`` does with a
        real id, here keyed by rounded coordinates instead.
        """
        if not self._dirty and self._outer_vertices is not None:
            return
        if self._outer_stale or self._outer_vertices is None:
            verts = self._compute_outer_vertices()
            new_hashes = [_vertex_hash(v) for v in verts]
            old_index_of_hash = (
                {h: i for i, h in enumerate(self._vertex_hashes)} if self._vertex_hashes is not None else {}
            )
            if len(verts):
                closest = np.zeros((len(verts), self.k))
                normals = np.zeros((len(verts), self.k))
                to_solve = []
                for i, h in enumerate(new_hashes):
                    old_i = old_index_of_hash.get(h)
                    if old_i is not None:
                        closest[i] = self._closest_points[old_i]
                        normals[i] = self._closest_normals[old_i]
                    else:
                        to_solve.append(i)
                if to_solve:
                    results = list(self.executor.map(
                        lambda i: self.projection_fn(verts[i], self.inner_vertices, norm=self.norm), to_solve
                    ))
                    for i, (c, n) in zip(to_solve, results):
                        closest[i] = c
                        normals[i] = n
            else:
                closest = np.zeros((0, self.k))
                normals = np.zeros((0, self.k))
            self._outer_vertices = verts
            self._vertex_hashes = new_hashes
            self._closest_points = closest
            self._closest_normals = normals
            self._outer_stale = False
        self._gap = gaps(self._outer_vertices, self._closest_points, norm=self.norm)
        self._dirty = False

    def find_trial_point(self) -> tuple[np.ndarray | None, float]:
        self._refresh()
        if len(self._outer_vertices) == 0:
            return None, 0.0
        i = int(np.argmax(self._gap))
        return self._outer_vertices[i], float(self._gap[i])

    def max_gap(self) -> float:
        self._refresh()
        if len(self._outer_vertices) == 0:
            return 0.0
        return float(self._gap.max())
