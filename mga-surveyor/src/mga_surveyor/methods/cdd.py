"""``CddMethod``.

- Uses cdd's (Fukuda's) Double Description method for the outer polytope's
  vertex set.
- Two arithmetic backends, selected via ``backend``: ``'float'`` (default)
  uses cdd's native C double arithmetic; ``'gmp'`` uses ``cdd.gmp``'s exact
  rational (GMP) arithmetic instead, at higher per-call cost, immune to the
  float backend's documented numerical-inconsistency failure mode on
  degenerate/near-degenerate accumulated halfspaces (see
  ``disdyakis_triacontahedron.py``'s module docstring for a real instance
  of that failure). Every halfspace coefficient is converted to
  ``fractions.Fraction`` before being handed to ``cdd.gmp`` -- an exact,
  lossless reinterpretation of whatever binary float value it already was,
  not a new source of imprecision, and not an attempt to recover exactness
  the upstream float computation (LP duals, etc.) never had in the first
  place.
- :meth:`CddMethod.add_cut` folds a new halfspace in incrementally --
  resuming cddlib's own Double Description computation instead of
  recomputing the whole polytope -- via ``polyhedron_append_halfspaces``
  (present on both the ``cdd`` and ``cdd.gmp`` modules), when the installed
  ``cdd``/``cdd.gmp`` has it. That function is a local patch on top of
  pycddlib/cddlib (vendored, with the patch, under ``vendor/cdd`` -- see
  ``vendor/cdd/README.md``) that isn't part of any upstream pycddlib
  release: cddlib's own public incremental-looking
  entry point, ``dd_DDInputAppend``, is an admitted stub ("This is
  incomplete. It simply solves the problem from scratch.") that just
  reruns the whole Double Description method. When
  ``polyhedron_append_halfspaces`` either isn't present (a stock
  ``pycddlib`` install) or raises (its own restrictions on the append
  aren't met for this polytope -- see its docstring), :meth:`_vertices_with_ids`
  transparently falls back to a full rebuild from ``self.outer_halfspaces``
  -- still correct either way, just not incremental for that one step. So
  this class keeps the strongest correctness guarantee of the two
  methods (cdd is a mature, independent implementation) regardless of
  which ``cdd`` build is installed; only the cost profile changes.
- The per-vertex closest-inner-point solve is cached across refreshes by
  generator identity: ``copy_generators_with_ids`` (also part of the local
  patch) tags every vertex with a persistent integer id that survives an
  incremental append unchanged, so :meth:`_refresh` can tell "this is the
  same vertex as last time" by exact id match instead of geometric
  comparison, reuse its cached closest point/normal, and only solve for
  vertices whose id is new. A full rebuild's ids are NOT comparable to a
  previous build's (cddlib restarts its id counter from scratch every
  time), so every vertex is treated as new whenever that path runs. When
  ``copy_generators_with_ids`` isn't available, every vertex is solved
  every refresh, exactly as before this caching existed.
"""
import time
from concurrent.futures import Executor
from fractions import Fraction
from types import ModuleType
from typing import Literal, Protocol

import numpy as np

from ..common import Halfspace, Norm, gaps, check_norm
from ..gurobi_projection import closest_point_on_hull
from .base import Method

#: Which cdd arithmetic backend to use -- ``'float'`` (cdd's native C
#: double arithmetic) or ``'gmp'`` (cdd.gmp's exact rational arithmetic).
CddBackend = Literal["float", "gmp"]


class ProjectionFn(Protocol):
    """Contract for `projection_fn`: project `point` onto `conv(points)`
    under `norm`, returning `(closest_point, normal)` -- exactly what
    `gurobi_projection.closest_point_on_hull` / `highs_projection.
    closest_point_on_hull` implement (see either's docstring).
    """
    def __call__(self, point: np.ndarray, points: np.ndarray, norm: Norm) -> tuple[np.ndarray, np.ndarray]: ...


def check_cdd_backend(backend: str) -> None:
    """Validate an untrusted ``backend`` value at a package boundary --
    the ``cdd``-specific counterpart of :func:`~mga_surveyor.common.check_norm`.
    """
    if backend not in ("float", "gmp"):
        raise ValueError(f"backend must be 'float' or 'gmp', got {backend!r}")


def _cdd_module(backend: CddBackend) -> ModuleType:
    """Import and return the cdd module for ``backend`` -- ``cdd`` itself
    for ``'float'``, ``cdd.gmp`` for ``'gmp'``. A separate function (rather
    than inlining the import) so every call site imports lazily, keeping
    ``cdd``/``cdd.gmp`` genuinely optional dependencies -- see the ``cdd``
    extra's note in ``pyproject.toml``.
    """
    check_cdd_backend(backend)
    if backend == "float":
        import cdd
        return cdd
    else:
        import cdd.gmp
        return cdd.gmp


def _halfspace_row(direction: np.ndarray, point: np.ndarray, backend: CddBackend) -> list:
    """One row of cdd's ``[b | A]`` H-representation for the halfspace
    ``{x : direction@x <= direction@point}``.

    ``direction`` is unit-normalized (divided by its own Euclidean norm)
    before the row is built -- scaling a halfspace's defining vector by any
    positive constant doesn't change the halfspace itself (``a@x<=b`` and
    ``(a/|a|)@x<=(b/|a|)`` are the same set), so this is free geometrically.
    It matters because cddlib's float-mode sign tests compare against one
    fixed ABSOLUTE tolerance (``dd_zero``, cddlib's own ``dd_almostzero``)
    -- if different cuts' rows come out at very different natural
    magnitudes (which they do here: `normal_to_normalized`'s per-dimension
    rescaling and this network's genuinely different-scale dimensions
    easily produce that), the same fixed tolerance is effectively far
    looser for large-magnitude rows and far tighter for small ones. Putting
    every row on the same (unit) scale removes that per-cut inconsistency.
    Confirmed empirically to bring float-mode's outer polytope into close
    agreement with gmp on this project's real network, where un-normalized
    float mode was badly wrong. A no-op for correctness under ``gmp``
    (exact arithmetic has no tolerance to be inconsistent about), so
    applying it unconditionally to both backends is safe.

    Under ``backend='gmp'``, every entry is converted to an exact
    ``Fraction`` -- required by ``cdd.gmp`` (it rejects plain ``float``
    input outright, since silently rounding would defeat the point of
    requesting exact arithmetic) and lossless: ``Fraction(x)`` on a float
    ``x`` is the exact rational value of that float's own binary
    representation, not a new approximation of it.
    """
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        # A zero direction is a degenerate/vacuous halfspace (0@x<=0@point,
        # trivially true everywhere) -- dividing by its own (zero) norm
        # would just turn it into a row of NaNs, which cddlib's linearity
        # detection can't handle at all (worse than the harmless all-zero
        # row this preserves instead).
        direction = direction / direction_norm
    b = float(direction @ np.asarray(point))
    row = [b] + [-float(c) for c in direction]
    if backend == "gmp":
        return [Fraction(x) for x in row]
    return row


def _extract_vertices(cdd_mod: ModuleType, poly) -> np.ndarray:
    """Pull the finite vertices (as opposed to rays) out of a cdd
    ``Polyhedron``'s generators -- a generator row is a vertex iff its
    leading column is ``1`` rather than ``0`` (cdd's homogeneous
    V-representation convention). Backend-agnostic: ``round()``/``float()``
    both work the same on a ``gmp`` row's ``Fraction`` entries as on a
    ``float`` row's plain floats.

    :param cdd_mod: The cdd module (``cdd`` or ``cdd.gmp``) ``poly`` was
        built with.
    :param poly: A cdd ``Polyhedron`` in H-representation with its
        generators already computed.
    :returns: The polyhedron's vertices, as an ``(m, k)`` array.
    :rtype: numpy.ndarray
    """
    gens = cdd_mod.copy_generators(poly)
    verts = [
        [float(x) for x in row[1:]]
        for row in gens.array
        if round(row[0]) == 1
    ]
    return np.array(verts)


def _extract_vertices_with_ids(cdd_mod: ModuleType, poly) -> tuple[np.ndarray, list[int] | None]:
    """Like :func:`_extract_vertices`, but also returns each vertex's
    persistent generator id -- via ``copy_generators_with_ids`` -- when the
    installed cdd module has that function and it accepts ``poly`` (it
    shares ``polyhedron_append_halfspaces``'s own restrictions: H-repr,
    full column rank; raises :exc:`ValueError` otherwise, e.g. for a
    not-yet-fully-bounded polytope). Returns ``ids=None`` whenever id
    tracking isn't available for this call, signalling the caller to fall
    back to solving every vertex fresh, same as if this function didn't
    exist at all.

    :param cdd_mod: The cdd module (``cdd`` or ``cdd.gmp``) ``poly`` was
        built with.
    :param poly: A cdd ``Polyhedron`` in H-representation with its
        generators already computed.
    :returns: ``(vertices, ids)`` -- ``vertices`` as an ``(m, k)`` array;
        ``ids`` a length-``m`` list aligned with its rows, or ``None``.
    :rtype: tuple
    """
    if not hasattr(cdd_mod, "copy_generators_with_ids"):
        return _extract_vertices(cdd_mod, poly), None
    try:
        gens, gen_ids = cdd_mod.copy_generators_with_ids(poly)
    except ValueError:
        return _extract_vertices(cdd_mod, poly), None
    verts = []
    ids = []
    for row, gid in zip(gens.array, gen_ids):
        if round(row[0]) == 1:
            verts.append([float(x) for x in row[1:]])
            ids.append(gid)
    return np.array(verts), ids


def cdd_vertices_of(outer_halfspaces: list[Halfspace], backend: CddBackend = "float") -> np.ndarray:
    """Ground-truth V-representation of the outer polytope defined by
    ``outer_halfspaces``, via a fresh cdd double description computation
    from scratch (all-at-once, not incremental).

    Used independently of any :class:`CddMethod` instance's own
    incremental state -- e.g. as a from-scratch ground-truth cross-check
    against a live instance's incrementally-maintained vertex set (see
    ``tests/test_cdd_incremental.py``), where reusing incremental state
    would defeat the point.

    :param outer_halfspaces: ``(direction, point)`` pairs, each defining a
        supporting hyperplane of the outer polytope.
    :param backend: ``'float'`` (default) or ``'gmp'`` -- see module
        docstring.
    :returns: The polytope's vertices, as an ``(m, k)`` array.
    :rtype: numpy.ndarray
    """
    cdd_mod = _cdd_module(backend)
    rows = [_halfspace_row(direction, point, backend) for direction, point in outer_halfspaces]
    mat = cdd_mod.matrix_from_array(rows, rep_type=cdd_mod.RepType.INEQUALITY)
    poly = cdd_mod.polyhedron_from_matrix(mat)
    return _extract_vertices(cdd_mod, poly)


class CddMethod(Method):
    """See module docstring.

    :param k: Dimension of the exploratory variables.
    :param norm: Distance metric, one of ``'l1'``, ``'l2'``, or ``'linf'``.
    :type norm: str
    :param executor: Used to run the independent per-vertex
        ``closest_point_on_hull`` solves concurrently after each
        refresh. A fresh ``ThreadPoolExecutor`` is created if none is
        given.
    :type executor: concurrent.futures.Executor, optional
    :param backend: ``'float'`` (default) or ``'gmp'`` -- see module
        docstring. Unrelated to ``projection_backend``/``projection_fn``
        below: this picks cdd's own outer-polytope arithmetic, not the
        inner-hull projection's solver.
    :type backend: str
    :param projection_fn: Lower-level override of the inner-hull projection
        callable itself, signature ``(point, points, norm) ->
        (closest_point, normal)`` -- takes precedence over
        ``projection_backend`` if given. Every call site goes through
        ``self.projection_fn``, so any callable matching that contract
        works, not just the two built-in backends.
    :param projection_backend: Which persistent solver to use for the
        inner-hull projection -- ``'gurobi'`` (default, all three norms) or
        ``'highs'`` (open-source, no license required; L1/Linf only --
        HiGHS has no quadratic-objective support). Same choice this
        project's now-removed incremental-vertex-enumeration method used
        to expose as its own ``backend`` parameter -- named differently
        here since ``backend`` on this class already means cdd's
        arithmetic.
    :type projection_backend: str
    :raises ValueError: If ``projection_backend`` is not ``'gurobi'`` or
        ``'highs'``.
    """

    def __init__(
        self, k: int, norm: Norm, executor: Executor | None = None, backend: CddBackend = "float",
        projection_fn: ProjectionFn | None = None, projection_backend: Literal["gurobi", "highs"] = "gurobi",
    ) -> None:
        check_norm(norm)
        check_cdd_backend(backend)
        super().__init__(k, norm)
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=8)
        self.executor = executor
        self.backend = backend
        self.projection_backend = projection_backend
        if projection_fn is not None:
            self.projection_fn = projection_fn
        elif projection_backend == "highs":
            # aliased on import: gurobi_projection.py/highs_projection.py both
            # name their projection function `closest_point_on_hull` -- see
            # either's own docstring -- so an alias is needed here to avoid
            # shadowing the one this module already imported for the
            # 'gurobi' default.
            from ..highs_projection import closest_point_on_hull as closest_point_on_hull_highs
            self.projection_fn = closest_point_on_hull_highs
        elif projection_backend == "gurobi":
            self.projection_fn = closest_point_on_hull
        else:
            raise ValueError(f"projection_backend must be 'gurobi' or 'highs', got {projection_backend!r}")

        self.inner_vertices = None
        self.outer_halfspaces = []
        self._poly = None  # persistent cdd.Polyhedron, folded into incrementally by add_cut; None until first built
        self._pending_rows = []  # halfspace rows from add_cut not yet folded into self._poly
        self._outer_vertices = None
        self._vertex_ids = None  # generator ids aligned with self._outer_vertices, or None if unsupported
        self._closest_points = None
        self._closest_normals = None  # aligned with self._outer_vertices; None whenever self._vertex_ids is None
        self._gap = None
        self._dirty = True
        self.stats: dict[str, float | int | None] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the instrumentation counters in :attr:`stats` -- purely
        additive bookkeeping, no effect on any other method's behavior or
        return value. :meth:`_refresh`/:meth:`_refresh_stale_closest_points`
        ACCUMULATE into these on every call (so several ``add_cut``s folded
        into one refresh, or a stale-point check plus a refresh, both add
        up rather than overwrite); call this to scope the running totals to
        whatever the caller considers one logical "iteration" (e.g. one
        `add_point` + its `add_cut`(s) + one `max_gap()`), then read
        :attr:`stats` after that iteration's calls are done.

        - ``cdd_time``/``projection_time``: seconds spent inside cdd's own
          append/rebuild call, and inside ``self.projection_fn`` solves,
          respectively (the latter summed across both
          :meth:`_refresh_stale_closest_points` and :meth:`_refresh`'s own
          solve step).
        - ``n_total``: the outer vertex count as of the last
          :meth:`_refresh` (a snapshot, not a delta -- last write wins even
          across several refreshes since the last reset).
        - ``n_new``/``n_removed``: vertices whose generator id appeared/
          disappeared since the previous refresh's id set. ``None``
          instead of an int whenever id tracking itself isn't available at
          all for this cdd install (see module docstring) -- distinct from
          ``0``, which means id tracking IS available and nothing actually
          changed. A full-rebuild fallback (id numbering restarts, so no
          previous id is comparable) counts as every previous vertex
          removed and every current one new, not a bug.
        - ``n_stale_checked``/``n_stale_failed``: from
          :meth:`_refresh_stale_closest_points`'s own cache-validity check
          -- how many previously-cached closest-point/normal pairs were
          checked against a newly added inner point, and how many of those
          failed (needed re-solving).
        - ``n_stale_failed_improved``: of the ones that failed and got
          re-solved, how many landed on a STRICTLY smaller gap than their
          stale cached value (rather than an equal one, e.g. a tie at the
          solver's own tolerance).
        """
        self.stats = {
            "cdd_time": 0.0,
            "projection_time": 0.0,
            "n_total": 0,
            "n_new": 0,
            "n_removed": 0,
            "n_stale_checked": 0,
            "n_stale_failed": 0,
            "n_stale_failed_improved": 0,
        }

    def bootstrap(self, initial_points: list[np.ndarray], initial_halfspaces: list[Halfspace]) -> None:
        self.inner_vertices = np.array(initial_points, dtype=float)
        self.outer_halfspaces = list(initial_halfspaces)
        self._poly = None
        self._pending_rows = []
        self._dirty = True

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
        """After the inner approximation grows, some cached closest-point/
        normal pairs (keyed by generator id -- see :meth:`_refresh`) may no
        longer be optimal -- exactly those whose cached separating
        hyperplane ``new_inner_vertex`` violates. Recompute only those
        (cheap ``O(outer_vertices)`` check first). Only called when
        ``self._closest_normals`` is populated, i.e. the last refresh
        actually had id-tracked caching to keep valid -- with no such
        cache, the next :meth:`_refresh` recomputes every vertex
        unconditionally anyway, so doing this check would be pure waste.

        :param new_inner_vertex: The newly added inner vertex.
        """
        if self._outer_vertices is None or len(self._outer_vertices) == 0:
            return
        rhs = new_inner_vertex[None, :] - self._closest_points
        still_valid = np.einsum("ij,ij->i", self._closest_normals, rhs) <= 1e-9
        stale_positions = np.where(~still_valid)[0]
        self.stats["n_stale_checked"] += len(still_valid)
        self.stats["n_stale_failed"] += len(stale_positions)
        if len(stale_positions) == 0:
            return
        stale_points = self._outer_vertices[stale_positions]
        old_closest = self._closest_points[stale_positions].copy()
        t0 = time.perf_counter()
        results = list(self.executor.map(
            lambda p: self.projection_fn(p, self.inner_vertices, norm=self.norm), stale_points
        ))
        self.stats["projection_time"] += time.perf_counter() - t0
        new_closest = np.array([c for c, _ in results])
        old_gap = gaps(stale_points, old_closest, norm=self.norm)
        new_gap = gaps(stale_points, new_closest, norm=self.norm)
        self.stats["n_stale_failed_improved"] += int(np.sum(new_gap < old_gap - 1e-12))
        for pos, (c, n) in zip(stale_positions.tolist(), results):
            self._closest_points[pos] = c
            self._closest_normals[pos] = n

    def add_cut(self, direction: np.ndarray, point: np.ndarray) -> None:
        direction = np.asarray(direction, dtype=float)
        point = np.asarray(point, dtype=float)
        self.outer_halfspaces = self.outer_halfspaces + [(direction, point)]
        self._pending_rows.append(_halfspace_row(direction, point, self.backend))
        self._dirty = True

    def _vertices_with_ids(self) -> tuple[np.ndarray, list[int] | None, bool]:
        """The outer polytope's current vertex set, with each vertex's
        persistent generator id when available.

        Appends whatever :meth:`add_cut` has queued in
        ``self._pending_rows`` onto the persistent ``self._poly`` via
        ``polyhedron_append_halfspaces`` (resuming cddlib's Double
        Description computation) when that function is available and
        succeeds; rebuilds ``self._poly`` from the complete
        ``self.outer_halfspaces`` from scratch otherwise -- on the first
        call, when the installed cdd module predates the
        incremental-append patch, or when the append itself raised (its
        own restrictions -- see its docstring -- weren't met for this
        polytope). Either path leaves ``self._poly`` holding the correct,
        fully up to date polyhedron.

        :returns: ``(vertices, ids, is_fresh_build)``. ``is_fresh_build``
            is ``True`` whenever a rebuild ran this call -- cddlib restarts
            its id numbering from scratch on every fresh
            ``dd_DDMatrix2Poly``, so ``ids`` (if not ``None``) are NOT
            comparable to whatever :class:`CddMethod` cached from a
            previous call in that case; the caller must treat every vertex
            as new rather than attempt id-based reuse.
        :rtype: tuple
        """
        cdd_mod = _cdd_module(self.backend)

        def full_rebuild() -> tuple[np.ndarray, list[int] | None]:
            rows = [_halfspace_row(d, p, self.backend) for d, p in self.outer_halfspaces]
            mat = cdd_mod.matrix_from_array(rows, rep_type=cdd_mod.RepType.INEQUALITY)
            self._poly = cdd_mod.polyhedron_from_matrix(mat)
            self._pending_rows = []
            return _extract_vertices_with_ids(cdd_mod, self._poly)

        if self._poly is None or not hasattr(cdd_mod, "polyhedron_append_halfspaces"):
            verts, ids = full_rebuild()
            return verts, ids, True

        if self._pending_rows:
            mat = cdd_mod.matrix_from_array(self._pending_rows, rep_type=cdd_mod.RepType.INEQUALITY)
            try:
                cdd_mod.polyhedron_append_halfspaces(self._poly, mat)
            except RuntimeError:
                verts, ids = full_rebuild()
                return verts, ids, True
            self._pending_rows = []

        verts, ids = _extract_vertices_with_ids(cdd_mod, self._poly)
        return verts, ids, False

    def _refresh(self) -> None:
        """Recompute the cached outer vertex set, per-vertex closest inner
        points, and per-vertex gap if stale, i.e. if ``self._dirty`` is set
        (by :meth:`bootstrap`, :meth:`add_point`, or :meth:`add_cut` since
        the last refresh); a no-op otherwise. Called at the top of every
        :meth:`find_trial_point` / :meth:`max_gap`, so back-to-back calls
        to either (with no state change in between) reuse this same cached
        `_gap` array instead of recomputing it from `_outer_vertices`/
        `_closest_points` each time.

        Solves ``closest_point_on_hull`` only for vertices whose generator
        id is new since the last refresh -- see :meth:`_vertices_with_ids`
        and the module docstring -- reusing the cached closest
        point/normal for every id that survived unchanged. Falls back to
        solving every vertex (the pre-caching behavior) whenever ids
        aren't available at all, or this call's vertex set came from a
        fresh rebuild whose ids aren't comparable to what was cached.

        Also accumulates into :attr:`stats` -- see :meth:`reset_stats`.
        """
        if not self._dirty and self._outer_vertices is not None:
            return
        prev_ids = self._vertex_ids
        t0 = time.perf_counter()
        verts, ids, is_fresh = self._vertices_with_ids()
        self.stats["cdd_time"] += time.perf_counter() - t0

        can_reuse = ids is not None and not is_fresh and prev_ids is not None
        if can_reuse:
            old_index_of_id = {vid: i for i, vid in enumerate(prev_ids)}
            new_id_set = set(ids)
            self.stats["n_new"] += sum(1 for vid in new_id_set if vid not in old_index_of_id)
            self.stats["n_removed"] += sum(1 for vid in prev_ids if vid not in new_id_set)
            closest = np.zeros((len(verts), self.k))
            normals = np.zeros((len(verts), self.k))
            to_solve = [i for i, vid in enumerate(ids) if vid not in old_index_of_id]
            for i, vid in enumerate(ids):
                old_i = old_index_of_id.get(vid)
                if old_i is not None:
                    closest[i] = self._closest_points[old_i]
                    normals[i] = self._closest_normals[old_i]
            if to_solve:
                t1 = time.perf_counter()
                results = list(self.executor.map(
                    lambda i: self.projection_fn(verts[i], self.inner_vertices, norm=self.norm), to_solve
                ))
                self.stats["projection_time"] += time.perf_counter() - t1
                for i, (c, n) in zip(to_solve, results):
                    closest[i] = c
                    normals[i] = n
        elif len(verts):
            if ids is not None:
                # A fresh rebuild's ids aren't comparable to prev_ids at all
                # (cddlib restarts id numbering) -- every current vertex
                # counts as new, every previous one as removed.
                self.stats["n_new"] += len(verts)
                self.stats["n_removed"] += len(prev_ids) if prev_ids is not None else 0
            else:
                self.stats["n_new"] = None
                self.stats["n_removed"] = None
            t1 = time.perf_counter()
            results = list(self.executor.map(
                lambda v: self.projection_fn(v, self.inner_vertices, norm=self.norm), verts
            ))
            self.stats["projection_time"] += time.perf_counter() - t1
            closest = np.array([r[0] for r in results])
            normals = np.array([r[1] for r in results]) if ids is not None else None
        else:
            if ids is not None:
                self.stats["n_removed"] += len(prev_ids) if prev_ids is not None else 0
            else:
                self.stats["n_new"] = None
                self.stats["n_removed"] = None
            closest = np.zeros((0, self.k))
            normals = np.zeros((0, self.k)) if ids is not None else None

        self.stats["n_total"] = len(verts)
        self._outer_vertices = verts
        self._vertex_ids = ids
        self._closest_points = closest
        self._closest_normals = normals
        self._gap = gaps(verts, closest, norm=self.norm)
        self._dirty = False

    def find_trial_point(self) -> tuple[np.ndarray | None, float]:
        self._refresh()
        if len(self._outer_vertices) == 0:
            return None, 0.0
        i = int(np.argmax(self._gap))
        return self._outer_vertices[i], float(self._gap[i])

    def find_trial_points_above_tol(self, tol: float) -> tuple[np.ndarray, np.ndarray]:
        """Like :meth:`find_trial_point`, but returns every outer vertex
        whose gap is at least `tol` (not just the single worst one),
        together with each one's own gap -- the first stage of solving
        several candidate closest-point targets in parallel instead of
        just one per iteration (see :mod:`mga_surveyor.explore`'s own
        docstring for why the single-target version can't be batched
        as-is: each iteration's target currently depends on the previous
        one's result).

        `tol` uses the same cutoff `explore.drive`'s own main loop applies
        before ever dispatching a solve for the single worst point:
        `gap < tol` means converged there, so `gap >= tol` (not the
        stricter `gap > tol`) is the matching "still worth solving"
        condition here -- a candidate right at `tol` exactly is still one
        `explore.drive` itself would have dispatched a solve for.

        Not (yet) part of the `Method` interface -- `explore.drive` still
        only ever calls `find_trial_point`. `BilevelMethod` has no
        analogous notion of "every other vertex above tol" to offer (its
        MILP reformulation finds the single worst point directly, without
        ever enumerating a vertex set), so this lives on `CddMethod` alone
        for now rather than on the shared base class.

        :param tol: The same convergence tolerance `explore.drive` itself
            is called with. No default -- see `explore.drive`'s own
            docstring on why it takes no default either.
        :returns: `(points, gaps)`, sorted by gap descending -- index 0 is
            the same point/gap `find_trial_point` itself would return.
            Both empty if the outer polytope has no vertices yet, or none
            are at least `tol`.
        """
        self._refresh()
        if len(self._outer_vertices) == 0:
            return np.empty((0, self.k)), np.empty(0)
        above_tol = np.where(self._gap >= tol)[0]
        if len(above_tol) == 0:
            return np.empty((0, self.k)), np.empty(0)
        idx = above_tol[np.argsort(self._gap[above_tol])[::-1]]
        return self._outer_vertices[idx], self._gap[idx]

    def find_pool_trial_points(
        self, max_parallel: int, tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched stand-in for :meth:`find_trial_point`: up to
        `max_parallel` above-`tol` candidate trial points to solve in
        parallel instead of just the single worst one -- the `max_parallel`
        worst ones, since :meth:`find_trial_points_above_tol` already
        returns them sorted by gap descending.

        Not (yet) wired into `explore.drive` -- that still only ever calls
        `find_trial_point`. This exists so an experimental batched driving
        loop has something to call; the sequential-by-construction
        original is untouched.

        :param max_parallel: Upper bound on how many points are returned
            -- meant to match however many parallel workers will actually
            solve them.
        :param tol: Passed straight to `find_trial_points_above_tol`. No
            default, for the same reason it has none there.
        :returns: `(points, gaps)`, capped at `max_parallel`, both empty
            if `find_trial_points_above_tol` itself returns nothing.
        """
        above_tol_points, above_tol_gaps = self.find_trial_points_above_tol(tol)
        return above_tol_points[:max_parallel], above_tol_gaps[:max_parallel]

    def max_gap(self) -> float:
        self._refresh()
        if len(self._outer_vertices) == 0:
            return 0.0
        return float(self._gap.max())
