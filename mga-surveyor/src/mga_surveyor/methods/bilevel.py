"""``BilevelMethod`` -- "bilevel solution".

- Implements the method of Turan, Moret & Bardow, "ORACLE: A rigorous
  metric and method to explore all near-optimal designs for energy
  systems", Computers & Chemical Engineering (2026).
- Only ``norm='linf'`` is supported; L1/L2 aren't implemented (use
  :class:`~mga_surveyor.methods.cdd.CddMethod` instead).
- Requires ``gurobipy`` and a Gurobi license; no open-source backend.
- Reformulates the max-min bilevel LP into a single-level MILP via KKT +
  Big-M (Fortuny-Amat/McCarl) linearization of the complementarity
  conditions.
"""
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

from ..common import Halfspace, Norm, check_norm
from .base import Method


def compute_safe_big_m(
    A_O: np.ndarray, b_O: np.ndarray, inner_points: np.ndarray, margin: float = 1.0,
) -> tuple[float, float]:
    """Derive provably-valid ``(M_P, M_D)`` directly from this problem's own
    data, instead of guessing fixed constants -- see
    :func:`solve_farthest_outer_point_milp`'s own docstring for what each
    bounds. Both constants follow from the KKT system's own structure:

    - ``M_P`` bounds ``w[i]`` and the two slacks
      ``slack_plus[j] = t - zO[j] + (Iw)[j]``,
      ``slack_minus[j] = t + zO[j] - (Iw)[j]``. ``w[i] <= 1`` always
      (convex weights). For the slacks: let ``lo[j]``/``hi[j]`` be the min/
      max of ``zO[j]`` over ``O`` (found here via ``2*nz`` tiny LPs on
      ``A_O``/``b_O``), and ``L[j]``/``U[j]`` the min/max of
      ``inner_points[:, j]`` (a plain array reduction -- any convex
      combination's ``j``-th coordinate is provably within
      ``[L[j], U[j]]``). Then ``t* = max_zO min_w ||zO - Iw||_inf <= T_MAX
      := max_j max(hi[j]-L[j], U[j]-lo[j])`` (how far ``O`` can possibly
      get from ``conv(inner_points)`` on any single axis), and each slack
      is at most ``t* + T_MAX <= 2*T_MAX``.
    - ``M_D`` bounds ``a_plus[j]``, ``a_minus[j]``, ``b_dual[i]``. The
      stationarity constraint ``sum(a_plus[j]+a_minus[j]) == 1`` (all
      terms >= 0) already forces ``a_plus[j] <= 1`` and ``a_minus[j] <=
      1`` on its own. At the complementary-slackness-active ``i`` (some
      ``w[i] > 0`` always exists, and ``b_dual[i] = 0`` there), the
      stationarity-in-``w`` equation pins ``eta`` to
      ``sum_j inner_points[i,j]*(a_plus[j]-a_minus[j])``, bounded by
      ``R_IN := max_{i,j} |inner_points[i,j]|`` since the ``a`` terms sum
      to 1 in absolute value. Propagating back through that same equation
      for every ``i`` gives ``b_dual[i] <= 2*R_IN``.

    Both are LOWER bounds on what's actually needed (exact at the true
    optimum) -- ``margin`` (default ``1.0``, i.e. no slack) inflates them
    for extra numerical headroom at the cost of a proportionally weaker
    relaxation; pass e.g. ``1.05`` if solves ever come back infeasible due
    to floating-point roundoff at the boundary.

    :param A_O: Outer polytope constraint matrix, with ``O`` given by
        ``A_O @ z <= b_O``.
    :param b_O: Outer polytope right-hand side.
    :param inner_points: Points whose convex hull is the inner
        approximation.
    :param margin: Multiplicative safety factor applied to both bounds.
    :returns: ``(M_P, M_D)``, ready to pass straight into
        :func:`solve_farthest_outer_point_milp`/:class:`BilevelMethod`.
    :rtype: tuple
    :raises ImportError: If ``gurobipy`` is not installed.
    """
    if gp is None:
        raise ImportError(
            "compute_safe_big_m requires the `gurobipy` package and a valid "
            "Gurobi license, same as solve_farthest_outer_point_milp itself."
        )
    inner_points = np.asarray(inner_points, dtype=float)
    nz = inner_points.shape[1]
    L = inner_points.min(axis=0)
    U = inner_points.max(axis=0)
    R_IN = float(np.max(np.abs(inner_points)))

    box_model = gp.Model()
    box_model.Params.OutputFlag = 0
    z = box_model.addVars(nz, lb=-GRB.INFINITY, name="z")
    for r in range(A_O.shape[0]):
        box_model.addConstr(gp.quicksum(A_O[r, j] * z[j] for j in range(nz)) <= b_O[r])
    lo = np.empty(nz)
    hi = np.empty(nz)
    for j in range(nz):
        box_model.setObjective(z[j], GRB.MAXIMIZE)
        box_model.optimize()
        hi[j] = z[j].X
        box_model.setObjective(z[j], GRB.MINIMIZE)
        box_model.optimize()
        lo[j] = z[j].X

    t_max = float(np.max(np.maximum(hi - L, U - lo)))
    M_P = max(1.0, 2.0 * t_max) * margin
    M_D = max(1.0, 2.0 * R_IN) * margin
    return M_P, M_D


def solve_farthest_outer_point_milp(
    A_O: np.ndarray, b_O: np.ndarray, inner_points: np.ndarray, time_limit: float | None,
    M_P: float | None = None, M_D: float | None = None, verbose: bool = False,
    diagnostics: dict | None = None,
    mip_gap: float = 1e-7, mip_gap_abs: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve ``max_{zO in O} min_{zI in conv(inner_points)} ||zO - zI||_inf``
    via the KKT-based single-level MILP reformulation. ``O`` is given as
    ``A_O @ z <= b_O``.

    :param A_O: Outer polytope constraint matrix, with ``O`` given by
        ``A_O @ z <= b_O``.
    :param b_O: Outer polytope right-hand side, with ``O`` given by
        ``A_O @ z <= b_O``.
    :param inner_points: Points whose convex hull is the inner
        approximation.
    :param time_limit: Seconds, forwarded to Gurobi's own ``TimeLimit``
        param -- no default, an explicit choice is required every call
        (``None`` for unbounded/exact-only, matching the original
        behavior before this parameter existed). ``None`` requires
        ``GRB.OPTIMAL``, raising on anything else. When set, a solve
        that hits the time limit with at least one feasible incumbent
        (``GRB.TIME_LIMIT``, ``model.SolCount > 0``) is ALSO accepted --
        the returned ``t_star`` is then a valid lower bound on the true
        max-min distance, not necessarily the exact value (the Big-M
        reformulation's usual exactness caveat, now compounded by not
        having actually proven optimality either). Pass ``diagnostics``
        to tell which case happened. A time-limited solve with NO
        incumbent at all still raises, same as any other failure status.
    :param M_P: Big-M constant bounding the primal solution
        (Fortuny-Amat/McCarl linearization of the complementarity
        conditions); must be a valid (large enough) bound for the specific
        problem, or the reformulation is not exact -- there is no automatic
        way to check this from inside the MILP itself. ``None`` (default)
        computes a provably-valid bound via :func:`compute_safe_big_m` on
        this exact call's own ``A_O``/``b_O``/``inner_points`` instead of
        guessing a fixed constant.
    :param M_D: Big-M constant bounding the dual solution; same caveat --
        and same ``None`` default -- as ``M_P``.
    :param verbose: If True, let Gurobi print its own solve log.
    :param diagnostics: If given, populated in place with ``solve_time``,
        ``status`` (Gurobi's own status code), ``is_optimal`` (``True``
        only for ``GRB.OPTIMAL``), ``mip_gap`` (``model.MIPGap`` --
        meaningless/``inf``-ish if no incumbent, which can't happen here
        since that case raises instead), and ``node_count``. Doesn't
        change the return value; purely for a caller that wants to log
        solve quality without re-deriving it.
    :param mip_gap: Forwarded to Gurobi's own ``MIPGap`` (relative
        optimality gap) convergence parameter.
    :param mip_gap_abs: Forwarded to Gurobi's own ``MIPGapAbs`` (absolute
        optimality gap) convergence parameter. ``None`` (default) leaves
        Gurobi's own default in place -- unlike ``mip_gap``, this isn't
        set unconditionally, since ``0`` is itself a meaningful value
        (require an EXACT proof, not just within some absolute tolerance)
        that shouldn't be silently equivalent to "unset".
    :returns: ``(z_O_star, t_star, w_star)`` -- the worst point, the max-min
        L_inf distance (exact iff ``diagnostics["is_optimal"]``, when
        given), and the convex-combination weights of its closest inner
        point (``z_I_star = w_star @ inner_points``).
    :rtype: tuple
    :raises ImportError: If ``gurobipy`` is not installed.
    :raises RuntimeError: If the solve ends in any status other than
        ``GRB.OPTIMAL``, or (only when ``time_limit`` is set)
        ``GRB.TIME_LIMIT`` with at least one feasible incumbent.
    """
    if gp is None:
        raise ImportError(
            "BilevelMethod requires the `gurobipy` package and a valid "
            "Gurobi license -- its MILP reformulation is Gurobi-specific, with no "
            "backend option (unlike CddMethod's own projection_backend; see module "
            "docstring). Install gurobipy, or use CddMethod instead "
            "(projection_backend='highs' needs no Gurobi at all, for norm in "
            "('l1', 'linf'))."
        )
    inner_points = np.asarray(inner_points, dtype=float)
    m, nz = inner_points.shape
    n_rows = A_O.shape[0]

    if M_P is None or M_D is None:
        computed_M_P, computed_M_D = compute_safe_big_m(A_O, b_O, inner_points)
        if M_P is None:
            M_P = computed_M_P
        if M_D is None:
            M_D = computed_M_D

    model = gp.Model()
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.MIPGap = mip_gap
    if mip_gap_abs is not None:
        model.Params.MIPGapAbs = mip_gap_abs
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    zO = model.addVars(nz, lb=-GRB.INFINITY, name="zO")
    t = model.addVar(lb=0.0, name="t")
    w = model.addVars(m, lb=0.0, name="w")

    a_plus = model.addVars(nz, lb=0.0, name="a_plus")
    a_minus = model.addVars(nz, lb=0.0, name="a_minus")
    b_dual = model.addVars(m, lb=0.0, name="b_dual")
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")

    u_plus = model.addVars(nz, vtype=GRB.BINARY, name="u_plus")
    u_minus = model.addVars(nz, vtype=GRB.BINARY, name="u_minus")
    v_bin = model.addVars(m, vtype=GRB.BINARY, name="v")

    for r in range(n_rows):
        model.addConstr(gp.quicksum(A_O[r, j] * zO[j] for j in range(nz)) <= b_O[r])

    slack_plus, slack_minus = {}, {}
    for j in range(nz):
        model.addConstr(zO[j] - gp.quicksum(inner_points[i, j] * w[i] for i in range(m)) - t <= 0)
        slack_plus[j] = t - zO[j] + gp.quicksum(inner_points[i, j] * w[i] for i in range(m))

        model.addConstr(-zO[j] + gp.quicksum(inner_points[i, j] * w[i] for i in range(m)) - t <= 0)
        slack_minus[j] = t + zO[j] - gp.quicksum(inner_points[i, j] * w[i] for i in range(m))

    model.addConstr(gp.quicksum(w[i] for i in range(m)) == 1, name="sum_to_one")

    model.addConstr(gp.quicksum(a_plus[j] + a_minus[j] for j in range(nz)) == 1, name="stat_t")
    for i in range(m):
        model.addConstr(
            eta - b_dual[i] - gp.quicksum(inner_points[i, j] * (a_plus[j] - a_minus[j]) for j in range(nz)) == 0,
            name=f"stat_w_{i}",
        )

    for j in range(nz):
        model.addConstr(a_plus[j] <= u_plus[j] * M_D)
        model.addConstr(slack_plus[j] <= (1 - u_plus[j]) * M_P)
        model.addConstr(a_minus[j] <= u_minus[j] * M_D)
        model.addConstr(slack_minus[j] <= (1 - u_minus[j]) * M_P)
    for i in range(m):
        model.addConstr(b_dual[i] <= v_bin[i] * M_D)
        model.addConstr(w[i] <= (1 - v_bin[i]) * M_P)

    # tightening from Caratheodory + L_inf structure (optional but cheap)
    model.addConstr(gp.quicksum(1 - v_bin[i] for i in range(m)) <= nz + 1)
    model.addConstr(gp.quicksum(1 - v_bin[i] for i in range(m)) >= 1)
    model.addConstr(gp.quicksum(u_plus[j] + u_minus[j] for j in range(nz)) >= 1)
    model.addConstr(gp.quicksum(u_plus[j] + u_minus[j] for j in range(nz)) <= nz)

    model.setObjective(t, GRB.MAXIMIZE)
    model.optimize()

    timed_out_with_incumbent = (
        time_limit is not None and model.Status == GRB.TIME_LIMIT and model.SolCount > 0
    )
    if model.Status != GRB.OPTIMAL and not timed_out_with_incumbent:
        raise RuntimeError(f"MILP failed: status {model.Status}")

    if diagnostics is not None:
        diagnostics["solve_time"] = model.Runtime  # Gurobi's own wall-clock solve time, no need to time it ourselves
        diagnostics["status"] = model.Status
        diagnostics["is_optimal"] = model.Status == GRB.OPTIMAL
        diagnostics["mip_gap"] = model.MIPGap
        diagnostics["node_count"] = model.NodeCount

    z_O_star = np.array([zO[j].X for j in range(nz)])
    t_star = t.X
    w_star = np.array([w[i].X for i in range(m)])
    return z_O_star, t_star, w_star


class BilevelMethod(Method):
    """See module docstring.

    :param k: Dimension of the exploratory variables.
    :param norm: Must be ``'linf'`` -- see module docstring for why.
    :type norm: str
    :param time_limit: Forwarded to :func:`solve_farthest_outer_point_milp`
        on every call -- see its own docstring. No default -- an explicit
        choice is required (``None`` for unbounded/exact-only, matching
        the original behavior before this parameter existed); set a real
        limit to accept a possibly-non-exact (but time-bounded) gap
        instead of risking an unboundedly long single solve -- see
        :attr:`stats`' ``is_optimal`` for whether a given call's result
        was actually exact.
    :param M_P: Big-M primal bound, forwarded to
        :func:`solve_farthest_outer_point_milp` on every call -- ``None``
        (default) computes a fresh, provably-valid bound via
        :func:`compute_safe_big_m` on every solve instead of a fixed
        constant (cheap relative to the MILP solve itself -- see that
        function's own docstring).
    :param M_D: Big-M dual bound, forwarded to
        :func:`solve_farthest_outer_point_milp` on every call -- same
        ``None`` default as ``M_P``.
    :param verbose: If True, let Gurobi print its own solve log.
    :param mip_gap: Forwarded to :func:`solve_farthest_outer_point_milp`
        on every call -- see its own docstring.
    :param mip_gap_abs: Forwarded to :func:`solve_farthest_outer_point_milp`
        on every call -- see its own docstring.
    :raises NotImplementedError: If ``norm`` is not ``'linf'``.
    """

    def __init__(
        self, k: int, norm: Norm, time_limit: float | None,
        M_P: float | None = None, M_D: float | None = None, verbose: bool = False,
        mip_gap: float = 1e-7, mip_gap_abs: float | None = None,
    ) -> None:
        check_norm(norm)
        if norm != "linf":
            raise NotImplementedError(
                "BilevelMethod only supports norm='linf' -- L1/L2 would each need a "
                "different reformulation, not implemented. "
                "Use CddMethod or BBMethod for other norms."
            )
        super().__init__(k, norm)
        self.M_P = M_P
        self.M_D = M_D
        self.verbose = verbose
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.mip_gap_abs = mip_gap_abs

        self.inner_vertices = None
        self.outer_halfspaces = []
        self._dirty = True
        self._trial_point = None
        self._gap = 0.0
        self.stats: dict[str, float | int | bool | None] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero :attr:`stats` -- mirrors :meth:`CddMethod.reset_stats`'s
        role for a caller that wants to scope solve diagnostics to one
        logical iteration, though here there's at most one solve per
        :meth:`find_trial_point`/:meth:`max_gap` call to begin with (no
        cross-call accumulation needed), so this just resets to defaults.
        Populated from :func:`solve_farthest_outer_point_milp`'s own
        ``diagnostics`` out-param after any call that actually solved
        (an unchanged/no-op ``find_trial_point`` call, or one with no
        halfspaces yet, leaves the previous values in place instead of
        clearing them).
        """
        self.stats = {
            "solve_time": 0.0,
            "status": None,
            "is_optimal": None,
            "mip_gap": None,
            "node_count": None,
        }

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

    def _matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble ``(A_O, b_O)`` from ``self.outer_halfspaces`` for
        :func:`solve_farthest_outer_point_milp`.

        :returns: ``(A_O, b_O)`` such that ``O`` is ``A_O @ z <= b_O``.
        :rtype: tuple
        """
        A_O = np.array([direction for direction, _ in self.outer_halfspaces])
        b_O = np.array([direction @ point for direction, point in self.outer_halfspaces])
        return A_O, b_O

    def find_trial_point(self) -> tuple[np.ndarray | None, float]:
        """Cached: recomputes only if ``self._dirty`` is set (by
        :meth:`bootstrap`, :meth:`add_point`, or :meth:`add_cut` since the
        last call); a repeat call in between -- notably from
        :meth:`max_gap` -- skips the MILP solve and reuses the cached
        result.
        """
        if not self._dirty:
            return self._trial_point, self._gap
        if not self.outer_halfspaces:
            self._trial_point, self._gap = None, 0.0
        else:
            A_O, b_O = self._matrices()
            diagnostics = {}
            z_O_star, t_star, _w_star = solve_farthest_outer_point_milp(
                A_O, b_O, self.inner_vertices, M_P=self.M_P, M_D=self.M_D, verbose=self.verbose,
                time_limit=self.time_limit, diagnostics=diagnostics,
                mip_gap=self.mip_gap, mip_gap_abs=self.mip_gap_abs,
            )
            self.stats.update(diagnostics)
            self._trial_point, self._gap = z_O_star, float(t_star)
        self._dirty = False
        return self._trial_point, self._gap

    def max_gap(self) -> float:
        _point, gap = self.find_trial_point()
        return gap
