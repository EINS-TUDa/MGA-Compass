"""``CddMethod``.

- Recomputes the outer polytope's vertex set from scratch each time it's
  needed, via cdd's (Fukuda's) Double Description method, rather than
  maintaining it incrementally.
- The simplest of the three methods and the one with the strongest
  correctness guarantee (cdd is a mature, independent implementation), at
  the cost of redoing O(accumulated halfspaces) work on every call instead
  of folding in one cut at a time. See
  :class:`~mga_surveyor.methods.vertex_enum.VertexEnumMethod` for the
  incremental alternative, and
  :meth:`~mga_surveyor.methods.vertex_enum.VertexEnumMethod.resync` for
  using this method as a periodic ground-truth checkpoint on top of that
  faster path instead of using it as the primary engine.

``cdd``'s public API has no supported way to resume a previous computation:
``dd_DDInputAppend`` exists in cddlib with a signature suggesting exactly
that, but its actual implementation is an admitted stub ("This is
incomplete. It simply solves the problem from scratch.") that just reruns
the whole Double Description method -- see project notes. So every call
here genuinely does pay for a full recomputation.
"""
from concurrent.futures import Executor

import numpy as np

from ..common import Halfspace, Norm, gaps, check_norm
from ..gurobi_projection import closest_point_on_hull
from .base import Method


def cdd_vertices_of(outer_halfspaces: list[Halfspace]) -> np.ndarray:
    """Ground-truth V-representation of the outer polytope defined by
    ``outer_halfspaces``, via a fresh cdd float-mode double description
    computation.

    :param outer_halfspaces: ``(direction, point)`` pairs, each defining a
        supporting hyperplane of the outer polytope.
    :returns: The polytope's vertices, as an ``(m, k)`` array.
    :rtype: numpy.ndarray
    """
    import cdd

    rows = []
    for direction, point in outer_halfspaces:
        b = float(np.asarray(direction) @ np.asarray(point))
        rows.append([b] + [-float(c) for c in direction])
    mat = cdd.matrix_from_array(rows, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    gens = cdd.copy_generators(poly)
    verts = [
        [float(x) for x in row[1:]]
        for row in gens.array
        if round(row[0]) == 1
    ]
    return np.array(verts)


class CddMethod(Method):
    """See module docstring.

    :param k: Dimension of the exploratory variables.
    :param norm: Distance metric, one of ``'l1'``, ``'l2'``, or ``'linf'``.
    :type norm: str
    :param executor: Used to run the independent per-vertex
        ``closest_point_on_hull`` solves concurrently after each
        recomputation. A fresh ``ThreadPoolExecutor`` is created if none is
        given.
    :type executor: concurrent.futures.Executor, optional
    """

    def __init__(self, k: int, norm: Norm, executor: Executor | None = None) -> None:
        check_norm(norm)
        super().__init__(k, norm)
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=8)
        self.executor = executor

        self.inner_vertices = None
        self.outer_halfspaces = []
        self._outer_vertices = None
        self._closest_points = None
        self._gap = None
        self._dirty = True

    def bootstrap(self, initial_points: list[np.ndarray], initial_halfspaces: list[Halfspace]) -> None:
        self.inner_vertices = np.array(initial_points, dtype=float)
        self.outer_halfspaces = list(initial_halfspaces)
        self._dirty = True

    def add_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        is_new = np.linalg.norm(self.inner_vertices - point, axis=1).min() > 1e-6
        if is_new:
            self.inner_vertices = np.vstack([self.inner_vertices, point])
            self._dirty = True
        return is_new

    def add_cut(self, direction: np.ndarray, point: np.ndarray) -> None:
        direction = np.asarray(direction, dtype=float)
        point = np.asarray(point, dtype=float)
        self.outer_halfspaces = self.outer_halfspaces + [(direction, point)]
        self._dirty = True

    def _refresh(self) -> None:
        """Recompute the cached outer vertex set, per-vertex closest inner
        points, and per-vertex gap if stale, i.e. if ``self._dirty`` is set
        (by :meth:`bootstrap`, :meth:`add_point`, or :meth:`add_cut` since
        the last refresh); a no-op otherwise. Called at the top of every
        :meth:`find_trial_point` / :meth:`max_gap`, so back-to-back calls
        to either (with no state change in between) reuse this same cached
        `_gap` array instead of recomputing it from `_outer_vertices`/
        `_closest_points` each time.
        """
        if not self._dirty and self._outer_vertices is not None:
            return
        verts = cdd_vertices_of(self.outer_halfspaces)
        if len(verts):
            results = list(self.executor.map(
                lambda v: closest_point_on_hull(v, self.inner_vertices, norm=self.norm), verts
            ))
            closest = np.array([r[0] for r in results])
        else:
            closest = np.zeros((0, self.k))
        self._outer_vertices = verts
        self._closest_points = closest
        self._gap = gaps(verts, closest, norm=self.norm)
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
