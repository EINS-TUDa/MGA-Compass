"""The :class:`Method` interface shared by all three worst-point implementations
(vertex enumeration, bilevel MILP, cdd), so a driving loop (see
:mod:`mga_surveyor.runner`) can be written once against this interface and
parameterized by which method to use.

Every method owns exactly the state needed to maintain an inner
approximation ``I`` (a set of points, implicitly their convex hull) and an
outer approximation ``O`` (a growing list of supporting halfspaces) of some
region -- and nothing else. In particular, no method ever calls back into
whatever produced a point or a cut: :meth:`Method.add_point` /
:meth:`Method.add_cut` are pure state updates, so the whole package is
independent of any specific test example or system model. Finding new
points/cuts to feed in is the caller's job (see the generic loop in
:mod:`mga_surveyor.runner`, or drive a method directly with your own).
"""
from abc import ABC, abstractmethod

import numpy as np

from ..common import Halfspace, Norm


class Method(ABC):
    """Common interface for a worst-point / convergence-metric implementation.

    :param k: Dimension of the exploratory variables.
    :type k: int
    :param norm: Distance metric used by the inner-hull projection that
        drives both the convergence metric and the direction of the next
        cut. One of ``'l1'``, ``'l2'``, or ``'linf'``.
    :type norm: str
    """

    def __init__(self, k: int, norm: Norm) -> None:
        self.k = k
        self.norm = norm

    @abstractmethod
    def bootstrap(self, initial_points: list[np.ndarray], initial_halfspaces: list[Halfspace]) -> None:
        """Seed the inner and outer approximations.

        Must be called exactly once, before any :meth:`add_point`,
        :meth:`add_cut`, or :meth:`find_trial_point` call.

        :param initial_points: Feasible points (e.g. the cost-optimal
            design) to seed the inner approximation with.
        :param initial_halfspaces: ``(direction, point)`` pairs, each
            defining a supporting hyperplane known to hold for the true
            region (e.g. from element-wise upper/lower bounds), to seed the
            outer approximation with.
        """

    @abstractmethod
    def add_point(self, point: np.ndarray) -> bool:
        """Grow the inner approximation with a newly found feasible point.

        :param point: The feasible point to add.
        :returns: True if the inner approximation actually changed, False
            if ``point`` duplicates an existing inner vertex within
            tolerance.
        :rtype: bool
        """

    @abstractmethod
    def add_cut(self, direction: np.ndarray, point: np.ndarray):
        """Shrink the outer approximation by adding a new supporting
        halfspace ``{x : direction @ (x - point) <= 0}``.

        :param direction: Normal vector of the new supporting hyperplane.
        :param point: A point on the new supporting hyperplane.
        """

    @abstractmethod
    def find_trial_point(self) -> tuple[np.ndarray | None, float]:
        """Find the point in the outer approximation furthest (in
        ``self.norm``) from the inner approximation.

        :returns: ``(trial_point, gap)``, where ``gap`` is the max-min
            distance itself (the convergence metric). ``(None, 0.0)`` if
            the outer approximation has no vertices (only possible before
            :meth:`bootstrap`).
        :rtype: tuple
        """

    @abstractmethod
    def max_gap(self) -> float:
        """The current convergence metric: the largest distance from any
        tracked outer-approximation vertex to the inner approximation. An
        upper bound on the distance from either approximation to the true
        region.

        :rtype: float
        """
