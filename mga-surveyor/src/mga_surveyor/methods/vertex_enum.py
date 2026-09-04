"""``VertexEnumMethod``.

- Reduces the worst-point search to vertex enumeration of the outer
  approximation ``O`` plus an independent, embarrassingly parallel convex
  projection at each vertex, using the fact that ``max_{z in O} (convex
  function of z)`` is attained at a vertex of ``O``.
- Maintains ``O``'s vertex set incrementally: each
  :meth:`VertexEnumMethod.add_cut` call folds in exactly one new halfspace
  by walking the (removed, kept) vertex split it induces and computing the
  new crossing vertices, rather than re-deriving the whole vertex set from
  scratch every time (that full-recompute alternative is
  :class:`~mga_surveyor.methods.cdd.CddMethod`). A halfspace redundant
  w.r.t. the current outer polytope (removes no current vertex -- exactly
  equivalent to globally redundant, and stays redundant forever, not just
  now, as more cuts only shrink the polytope further) is detected and
  discarded rather than appended to ``outer_halfspaces``, so a run's
  worth of redundant cuts doesn't silently slow down every future
  :meth:`VertexEnumMethod.resync`.
- Assumes SIMPLE vertices (exactly ``k`` tight constraints each) for edge
  detection between removed/kept vertex pairs, carried over from the
  original implementation this was ported from; no degeneracy-prevention
  mechanism is implemented at this layer. Use
  :meth:`VertexEnumMethod.resync` periodically (backed by
  :class:`~mga_surveyor.methods.cdd.CddMethod`'s ground-truth computation)
  for a safety net against it.
"""
from concurrent.futures import Executor
from typing import Callable, Literal, Protocol

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import HalfspaceIntersection

from ..common import Halfspace, Norm, gaps, check_norm
from ..gurobi_projection import closest_point_on_hull
from .base import Method


class ProjectionFn(Protocol):
    """Contract for `projection_fn`: project `point` onto `conv(points)`
    under `norm`, returning `(closest_point, normal)` -- exactly what
    `gurobi_projection.closest_point_on_hull` / `highs_projection.
    closest_point_on_hull` implement (see either's docstring).
    """
    def __call__(self, point: np.ndarray, points: np.ndarray, norm: Norm) -> tuple[np.ndarray, np.ndarray]: ...


def _normalize_halfspace(halfspace: Halfspace) -> Halfspace:
    """Canonicalize a halfspace to unit-norm direction (``point`` is
    unchanged -- ``{x : direction@x <= direction@point}`` is invariant
    under scaling ``direction`` by any ``c > 0``, since both sides of the
    inequality scale together).

    Every stored halfspace's "distance to hyperplane" (``direction@x -
    offset``) then has consistent, comparable real-geometric units
    regardless of the raw coefficient scale its ``direction`` happened to
    arrive with -- box halfspaces, geometric MGA cuts, and
    objective-derived cuts have wildly different natural magnitudes in
    this package. Without this, ``self.tol``/``self.dedup_tol``
    comparisons (the kept-vs-removed vertex split in
    :meth:`VertexEnumMethod.add_cut`, and the active-constraint
    classification in :func:`_compute_active_constrs`) silently mean a
    different real-world distance for every different halfspace's own
    scale -- a large-coefficient halfspace's tolerance is effectively far
    tighter (in real units) than a small-coefficient one's, which
    misclassifies which constraints are genuinely tight at a vertex and
    was a real contributor to :meth:`VertexEnumMethod.add_cut`'s
    crossing-vertex count exploding on an already near-degenerate
    polytope (confirmed against the real network: one cut produced over
    50,000 candidate crossing vertices).

    :param halfspace: A ``(direction, point)`` supporting hyperplane.
    :returns: The same halfspace, with ``direction`` rescaled to unit
        L2 norm.
    :rtype: tuple
    """
    direction, point = halfspace
    direction = np.asarray(direction, dtype=float)
    point = np.asarray(point, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-300:
        # A zero-direction "halfspace" is degenerate (either the whole
        # space or nothing) regardless of scale -- nothing meaningful to
        # normalize by; pass it through unchanged rather than divide by
        # zero.
        return direction, point
    return direction / norm, point


def _halfspaces_to_scipy(outer_halfspaces: list[Halfspace]) -> np.ndarray:
    """Convert ``outer_halfspaces`` into the ``[A | b]`` row format expected
    by :class:`scipy.spatial.HalfspaceIntersection`, i.e. rows such that
    ``A @ x + b <= 0``. Only :meth:`VertexEnumMethod.bootstrap` needs this
    -- the one place this class builds an initial outer vertex set via
    ``HalfspaceIntersection`` rather than incremental edge-walking.

    :param outer_halfspaces: ``(direction, point)`` supporting hyperplanes.
    :returns: The stacked ``[A | b]`` rows.
    :rtype: numpy.ndarray
    """
    rows = []
    for direction, point in outer_halfspaces:
        offset = direction @ point
        rows.append(np.append(direction, -offset))
    return np.array(rows)


def _compute_active_constrs(
    outer_vertices: np.ndarray, outer_halfspaces: list[Halfspace], tol: float = 1e-7,
) -> list[list[int]]:
    """For each outer vertex, the indices of ``outer_halfspaces`` that are
    tight there (i.e. the vertex's active-constraint set) -- the
    bookkeeping :func:`_find_edges` needs to do incremental edge-walking in
    :meth:`VertexEnumMethod.add_cut`. Only ever (re)built wholesale in
    :meth:`VertexEnumMethod.bootstrap` and :meth:`VertexEnumMethod.resync`;
    :meth:`VertexEnumMethod.add_cut` extends it incrementally itself
    instead of calling this again.

    :param outer_vertices: The outer polytope's vertices.
    :param outer_halfspaces: ``(direction, point)`` supporting hyperplanes.
    :param tol: Tolerance for treating a constraint as tight at a vertex.
    :returns: For each vertex, the sorted list of its tight halfspace
        indices.
    :rtype: list[list[int]]
    """
    A = np.array([direction for direction, _ in outer_halfspaces])
    b = np.array([direction @ point for direction, point in outer_halfspaces])
    return [list(np.where(np.abs(A @ v - b) < tol)[0]) for v in outer_vertices]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All pairwise squared distances between rows of ``a`` and rows of
    ``b``, via ``||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y`` (one matmul, no
    ``O(n*m)`` Python loop and no ``O(n*m*k)`` intermediate array).

    :param a: An ``(n, k)`` array.
    :param b: An ``(m, k)`` array.
    :returns: The ``(n, m)`` matrix of squared distances.
    :rtype: numpy.ndarray
    """
    a_sq = np.sum(a * a, axis=1)
    b_sq = np.sum(b * b, axis=1)
    d = a_sq[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    np.maximum(d, 0, out=d)
    return d


def _dedupe_new_vertices(
    kept_verts: np.ndarray, kept_active: list[list[int]], kept_closest: np.ndarray, kept_normals: np.ndarray,
    new_verts: list[np.ndarray], new_active: list[list[int]], new_closest: list[np.ndarray], new_normals: list[np.ndarray],
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    """Merge geometrically-identical vertices among the newly created ones,
    and into an existing kept vertex if one matches (can happen when a new
    cut is tangent at a point reached via two different edges).

    ``kept_verts`` are assumed already pairwise-distinct, so only new-vs-new
    and new-vs-kept pairs are checked -- the kept-vs-kept comparison can be
    skipped safely.

    :param kept_verts: Vertices of the previous outer polytope that survive
        the new cut.
    :param kept_active: Active-constraint sets for ``kept_verts``.
    :param kept_closest: Cached closest inner points for ``kept_verts``.
    :param kept_normals: Cached closest-point normals for ``kept_verts``.
    :param new_verts: Newly created crossing vertices.
    :param new_active: Active-constraint sets for ``new_verts``.
    :param new_closest: Closest inner points for ``new_verts``.
    :param new_normals: Closest-point normals for ``new_verts``.
    :param tol: Distance tolerance below which two vertices are considered
        identical.
    :returns: ``(verts, active, closest, normals)`` -- the deduplicated,
        merged vertex set and its associated per-vertex state.
    :rtype: tuple
    """
    n_new = len(new_verts)
    kept_verts = np.asarray(kept_verts)
    result_verts = list(kept_verts)
    result_active = [list(a) for a in kept_active]
    result_closest = list(kept_closest)
    result_normals = list(kept_normals)

    if n_new == 0:
        return np.array(result_verts), result_active, np.array(result_closest), np.array(result_normals)

    new_verts = np.asarray(new_verts)
    match_new = _sq_dists(new_verts, new_verts) < tol * tol
    match_kept = _sq_dists(new_verts, kept_verts) < tol * tol if len(kept_verts) else np.zeros((n_new, 0), dtype=bool)

    parent = list(range(n_new))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    ii, jj = np.where(np.triu(match_new, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        union(a, b)

    cluster_to_kept = {}
    rows, cols = np.where(match_kept)
    for r, c in zip(rows.tolist(), cols.tolist()):
        cluster_to_kept.setdefault(find(r), c)

    clusters = {}
    for i in range(n_new):
        clusters.setdefault(find(i), []).append(i)

    for root, members in clusters.items():
        active_union = set()
        for idx in members:
            active_union |= set(new_active[idx])
        if root in cluster_to_kept:
            kidx = cluster_to_kept[root]
            result_active[kidx] = sorted(set(result_active[kidx]) | active_union)
        else:
            rep = members[0]
            result_verts.append(new_verts[rep])
            result_active.append(sorted(active_union))
            result_closest.append(new_closest[rep])
            result_normals.append(new_normals[rep])

    return np.array(result_verts), result_active, np.array(result_closest), np.array(result_normals)


def _find_edges(
    removed_idxs: np.ndarray, kept_idxs: np.ndarray, active_constrs: list[list[int]], k: int, n_cols: int,
) -> list[tuple[int, int, list[int]]]:
    """Vectorized replacement for the ``O(removed*kept)`` double loop: find
    all (removed, kept) pairs sharing exactly ``k-1`` active constraints via
    sparse boolean matrix multiplication, instead of per-pair set
    intersections.

    Assumes simple vertices (exactly ``k`` tight constraints each) -- see
    module docstring.

    :param removed_idxs: Indices of outer vertices cut off by the new
        halfspace.
    :param kept_idxs: Indices of outer vertices that survive the new
        halfspace.
    :param active_constrs: Each vertex's active-constraint index list.
    :param k: Dimension of the exploratory variables.
    :param n_cols: Number of halfspaces so far (including the new one).
    :returns: ``(removed_idx, kept_idx, sorted_shared_constrs)`` triples,
        one per edge crossing the new halfspace.
    :rtype: list[tuple]
    """
    removed_idxs = list(removed_idxs)
    kept_idxs = list(kept_idxs)
    if not removed_idxs or not kept_idxs:
        return []

    def build_sparse(idxs):
        if not idxs:
            return csr_matrix((0, n_cols), dtype=np.int32)
        lengths = np.fromiter((len(active_constrs[i]) for i in idxs), dtype=np.int64, count=len(idxs))
        indptr = np.zeros(len(idxs) + 1, dtype=np.int64)
        np.cumsum(lengths, out=indptr[1:])
        indices = np.concatenate([active_constrs[i] for i in idxs]) if indptr[-1] else np.array([], dtype=np.int64)
        data = np.ones(len(indices), dtype=np.int32)
        return csr_matrix((data, indices, indptr), shape=(len(idxs), n_cols))

    R = build_sparse(removed_idxs)
    K = build_sparse(kept_idxs)
    inter = (R @ K.T).toarray()
    a_idx, b_idx = np.where(inter == k - 1)

    edges = []
    for a, b in zip(a_idx, b_idx):
        ri = removed_idxs[a]
        ki = kept_idxs[b]
        shared = sorted(set(active_constrs[ri]) & set(active_constrs[ki]))
        edges.append((ri, ki, shared))
    return edges


class VertexEnumMethod(Method):
    """See module docstring.

    :param k: Dimension of the exploratory variables.
    :param norm: Distance metric, one of ``'l1'``, ``'l2'``, or ``'linf'``.
    :type norm: str
    :param executor: Used to run the independent per-vertex
        ``closest_point_on_hull`` solves concurrently. A fresh
        ``ThreadPoolExecutor`` is created if none is given.
    :type executor: concurrent.futures.Executor, optional
    :param tol: Tolerance used to classify an outer vertex as kept vs.
        removed by a new cut.
    :param dedup_tol: Distance tolerance for merging geometrically-identical
        vertices in :func:`_dedupe_new_vertices`, and for treating a new
        point as a duplicate inner vertex in :meth:`add_point`.
    :param resync_every: If set to a positive integer N, automatically call
        :meth:`resync` after every N :meth:`add_cut` calls, as a periodic
        ground-truth checkpoint against ``cdd`` -- see project notes on why
        this can matter (it found real, if rare, drift between the
        incrementally-maintained vertex set and the true outer polytope in
        stress testing). Disabled by default (``None``): the incremental
        algorithm has never been observed to need it at the problem sizes
        tested so far, and cdd's cost grows with the outer polytope's
        vertex count, so enabling it isn't free -- opt in deliberately,
        don't treat it as a default safety net.
    :type resync_every: int, optional
    :param projection_fn: Lower-level override of the inner-hull projection
        callable itself, signature ``(point, points, norm) ->
        (closest_point, normal)`` -- takes precedence over ``backend`` if
        given. Every call site goes through ``self.projection_fn``, so any
        callable matching that contract works, not just the two built-in
        backends.
    :param backend: Which persistent solver to use for the inner-hull
        projection -- ``'gurobi'`` (default, all three norms) or
        ``'highs'`` (open-source, no license required; L1/Linf only --
        HiGHS has no quadratic-objective support, confirmed separately, so
        it can't do L2). Measured comparably fast to (and sometimes faster
        than) Gurobi for this workload once the point set isn't tiny -- see
        project notes -- so ``'highs'`` is a real choice for L1/Linf, not
        just a license-free fallback.
    :type backend: str
    :raises ValueError: If ``backend`` is not ``'gurobi'`` or ``'highs'``.
    """

    def __init__(
        self, k: int, norm: Norm, executor: Executor | None = None, tol: float = 1e-6, dedup_tol: float = 1e-6,
        resync_every: int | None = None, projection_fn: ProjectionFn | None = None,
        backend: Literal["gurobi", "highs"] = "gurobi",
    ) -> None:
        check_norm(norm)
        super().__init__(k, norm)
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=8)
        self.executor = executor
        self.backend = backend
        if projection_fn is not None:
            self.projection_fn = projection_fn
        elif backend == "highs":
            # aliased on import: this module already bound the gurobi one to
            # the bare name `closest_point_on_hull` above (both backend
            # modules use that same name -- see gurobi_projection.py/
            # highs_projection.py docstrings -- so an alias is needed here to
            # avoid shadowing it).
            from ..highs_projection import closest_point_on_hull as closest_point_on_hull_highs
            self.projection_fn = closest_point_on_hull_highs
        elif backend == "gurobi":
            self.projection_fn = closest_point_on_hull
        else:
            raise ValueError(f"backend must be 'gurobi' or 'highs', got {backend!r}")
        self.tol = tol
        self.dedup_tol = dedup_tol
        self.resync_every = resync_every

        self.inner_vertices = None
        self.outer_halfspaces = []
        self.outer_vertices = None
        self.active_constrs = None
        self.closest_points = None
        self.closest_normals = None
        self._cut_count = 0
        self.resync_log = []  # (cut_count, n_only_mine, n_only_cdd) for every auto-resync performed

    def bootstrap(self, initial_points: list[np.ndarray], initial_halfspaces: list[Halfspace]) -> None:
        self.inner_vertices = np.array(initial_points, dtype=float)
        self.outer_halfspaces = [_normalize_halfspace(hs) for hs in initial_halfspaces]
        interior_point = self.inner_vertices.mean(axis=0)
        hs = HalfspaceIntersection(_halfspaces_to_scipy(self.outer_halfspaces), interior_point)
        self.outer_vertices = hs.intersections
        self.active_constrs = _compute_active_constrs(self.outer_vertices, self.outer_halfspaces)
        results = list(self.executor.map(
            lambda v: self.projection_fn(v, self.inner_vertices, norm=self.norm), self.outer_vertices
        ))
        self.closest_points = np.array([r[0] for r in results])
        self.closest_normals = np.array([r[1] for r in results])

    def add_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        is_new = np.linalg.norm(self.inner_vertices - point, axis=1).min() > self.dedup_tol
        if is_new:
            self.inner_vertices = np.vstack([self.inner_vertices, point])
            self._refresh_stale_closest_points(point)
        return is_new

    def _refresh_stale_closest_points(self, new_inner_vertex: np.ndarray) -> None:
        """After the inner approximation grows, some outer vertices' cached
        closest-point/normal may no longer be optimal -- exactly those
        whose cached separating hyperplane ``new_inner_vertex`` violates.
        Recompute only those (cheap ``O(outer_vertices)`` check first).

        :param new_inner_vertex: The newly added inner vertex.
        """
        if self.outer_vertices is None or len(self.outer_vertices) == 0:
            return
        rhs = new_inner_vertex[None, :] - self.closest_points
        still_valid = np.einsum("ij,ij->i", self.closest_normals, rhs) <= 1e-9
        stale_positions = np.where(~still_valid)[0]
        if len(stale_positions) == 0:
            return
        stale_points = list(self.outer_vertices[stale_positions])
        results = list(self.executor.map(
            lambda p: self.projection_fn(p, self.inner_vertices, norm=self.norm), stale_points
        ))
        for pos, (c, n) in zip(stale_positions.tolist(), results):
            self.closest_points[pos] = c
            self.closest_normals[pos] = n

    def add_cut(self, direction: np.ndarray, point: np.ndarray) -> tuple[int, int]:
        """Fold in one new halfspace incrementally -- redundant w.r.t. the
        current outer polytope (removes no outer vertex) is detected and
        the halfspace is discarded rather than appended to
        ``outer_halfspaces``, in which case this is a cheap no-op:
        ``(0, 0)``.

        Detecting this is exact, not a heuristic: a halfspace containing
        every vertex of a polytope contains the polytope's whole convex
        hull too, so "removes no current vertex" and "globally redundant
        given the current outer_halfspaces" are the same statement. And
        it stays true forever, not just now -- redundant w.r.t. the
        current (larger) polytope implies redundant w.r.t. any future
        (smaller, further-cut) one, since the smaller polytope is even
        more contained in the halfspace. So discarding it permanently is
        safe, not just an optimization for this call. Without this,
        ``outer_halfspaces`` only ever grows, redundant entries included
        -- costing nothing per :meth:`add_cut` itself (the checks below
        already short-circuit cheaply either way), but making every
        future :meth:`resync` (a full cdd recomputation over the whole
        list) slower than it needs to be.

        :param direction: Normal vector of the new supporting hyperplane.
        :param point: A point on the new supporting hyperplane.
        :returns: ``(n_new_vertices, n_removed_vertices)`` -- how many
            outer vertices the new cut created and removed. ``(0, 0)``
            both when the cut was redundant and (degenerately) when it
            wasn't but happened to create as many vertices as it removed.
        :rtype: tuple
        """
        direction, point = _normalize_halfspace((direction, point))

        offset = direction @ point
        signed = self.outer_vertices @ direction - offset

        kept_idxs = np.where(signed <= self.tol)[0]
        removed_idxs = np.where(signed > self.tol)[0]

        if len(removed_idxs) == 0:
            return 0, 0

        new_idx = len(self.outer_halfspaces)
        kept_verts = self.outer_vertices[kept_idxs]
        # already up to date -- add_point refreshes these eagerly whenever
        # I grows, so no separate inner-changed check is needed here
        kept_closest = self.closest_points[kept_idxs]
        kept_normals = self.closest_normals[kept_idxs]
        kept_active = [self.active_constrs[i] for i in kept_idxs]

        edges = _find_edges(removed_idxs, kept_idxs, self.active_constrs, self.k, new_idx)
        crossing_verts = []
        crossing_active = []
        for ri, ki, shared in edges:
            t = signed[ri] / (signed[ri] - signed[ki])
            new_v = self.outer_vertices[ri] + t * (self.outer_vertices[ki] - self.outer_vertices[ri])
            crossing_verts.append(new_v)
            crossing_active.append(shared + [new_idx])

        if crossing_verts:
            results = list(self.executor.map(
                lambda p: self.projection_fn(p, self.inner_vertices, norm=self.norm), crossing_verts
            ))
        else:
            results = []

        crossing_closest = [r[0] for r in results]
        crossing_normals = [r[1] for r in results]

        self.outer_halfspaces = self.outer_halfspaces + [(direction, point)]
        self.outer_vertices, self.active_constrs, self.closest_points, self.closest_normals = _dedupe_new_vertices(
            kept_verts, kept_active, kept_closest, kept_normals,
            crossing_verts, crossing_active, crossing_closest, crossing_normals,
            tol=self.dedup_tol,
        )
        n_new_vertices = len(self.outer_vertices) - len(kept_idxs)
        n_removed_vertices = len(removed_idxs)

        self._cut_count += 1
        if self.resync_every and self._cut_count % self.resync_every == 0:
            n_only_mine, n_only_cdd = self.resync()
            if n_only_mine or n_only_cdd:
                self.resync_log.append((self._cut_count, n_only_mine, n_only_cdd))

        return n_new_vertices, n_removed_vertices

    def find_trial_point(self) -> tuple[np.ndarray | None, float]:
        if self.outer_vertices is None or len(self.outer_vertices) == 0:
            return None, 0.0
        gap = gaps(self.outer_vertices, self.closest_points, norm=self.norm)
        i = int(np.argmax(gap))
        return self.outer_vertices[i], float(gap[i])

    def max_gap(self) -> float:
        if self.outer_vertices is None or len(self.outer_vertices) == 0:
            return 0.0
        return float(gaps(self.outer_vertices, self.closest_points, norm=self.norm).max())

    def resync(self, cdd_vertices_fn: Callable[[list[Halfspace]], np.ndarray] | None = None) -> tuple[int, int]:
        """Periodic correctness checkpoint: recompute the outer polytope's
        vertex set from scratch via an independent implementation and
        replace the incrementally-maintained state with it if they
        disagree.

        :param cdd_vertices_fn: If given, called with
            ``self.outer_halfspaces`` and must return the true vertex set
            as an ``(m, k)`` array. Defaults to
            :func:`~mga_surveyor.methods.cdd.cdd_vertices_of`, i.e. cdd's
            ground-truth computation.
        :returns: ``(n_only_mine, n_only_cdd)``, both 0 if no discrepancy
            found.
        :rtype: tuple
        """
        if cdd_vertices_fn is None:
            from .cdd import cdd_vertices_of
            cdd_vertices_fn = cdd_vertices_of

        true_vertices = cdd_vertices_fn(self.outer_halfspaces)

        mine_set = {tuple(round(float(x), 6) for x in v) for v in self.outer_vertices}
        cdd_set = {tuple(round(float(x), 6) for x in v) for v in true_vertices}
        only_mine = mine_set - cdd_set
        only_cdd = cdd_set - mine_set
        if not only_mine and not only_cdd:
            return 0, 0

        self.outer_vertices = true_vertices
        self.active_constrs = _compute_active_constrs(true_vertices, self.outer_halfspaces, tol=self.tol)
        results = list(self.executor.map(
            lambda v: self.projection_fn(v, self.inner_vertices, norm=self.norm), true_vertices
        ))
        self.closest_points = np.array([r[0] for r in results])
        self.closest_normals = np.array([r[1] for r in results])
        return len(only_mine), len(only_cdd)
